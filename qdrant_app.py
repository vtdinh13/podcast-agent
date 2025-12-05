import ast
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import logfire
import streamlit as st
from jaxn import JSONParserHandler, StreamingJSONParser
from pydantic_ai.messages import FunctionToolCallEvent, ModelRequest, ModelResponse, TextPart

from habit_agent import SearchResultResponse, create_research_agent
from token_guard import activate_guard, TokenBudget, TokenBudgetExceeded

logfire.configure()


@st.cache_resource(show_spinner=False)
def load_agent():
    """Instantiate and cache the Qdrant-only agent across reruns."""
    return create_research_agent()


def parse_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _to_serializable(content: Any) -> Any:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if hasattr(content, "model_dump"):
        return content.model_dump()
    if hasattr(content, "dict"):
        return content.dict()
    if isinstance(content, (list, tuple)):
        return [_to_serializable(item) for item in content]
    if isinstance(content, dict):
        return {key: _to_serializable(value) for key, value in content.items()}
    return content


def _payload_to_json_text(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{'") or stripped.startswith("['"):
            try:
                reconstructed = ast.literal_eval(payload)
            except Exception:
                return payload
            serializable = _to_serializable(reconstructed)
            try:
                return json.dumps(serializable, ensure_ascii=False)
            except (TypeError, ValueError):
                return payload
        return payload

    serializable = _to_serializable(payload)
    try:
        return json.dumps(serializable, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def format_reference_line(ref: Any) -> str:
    if ref is None:
        return "Reference"
    if hasattr(ref, "model_dump"):
        ref = ref.model_dump()
    elif hasattr(ref, "dict"):
        ref = ref.dict()
    if not isinstance(ref, dict):
        try:
            return str(ref)
        except Exception:  # noqa: BLE001
            return "Reference"
    if ref.get("episode_name") is not None:
        episode = ref.get("episode_name")
        start = ref.get("start") or ref.get("start_time") or ""
        end = ref.get("end") or ref.get("end_time") or ""
        window = " - ".join([t for t in (start, end) if t])
        return f"{episode} ({window})" if window else episode

    title = ref.get("title") or "Unknown Title"
    url = ref.get("url") or ""
    return f"*{title}* {url}".strip()


class StreamlitSearchResultHandler(JSONParserHandler):
    """Stream SearchResultResponse output into the chat placeholder."""

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator) -> None:
        super().__init__()
        self.placeholder = placeholder
        self.description_md: str = ""
        self.sections_md: str = ""
        self.section_reference_paths: set[str] = set()
        self.rendered_text: str = ""

    def _render(self) -> None:
        blocks: List[str] = []
        if self.description_md.strip():
            blocks.append(self.description_md.strip())
        if self.sections_md.strip():
            blocks.append(self.sections_md.strip())
        text = "\n\n".join(blocks).strip()
        if not text:
            return
        self.placeholder.markdown(text)
        self.rendered_text = text

    def on_field_end(
        self, path: str, field_name: str, value: str, parsed_value: Any = None
    ) -> None:
        if field_name == "description":
            self.description_md = f"### Description\n\n{value}"
            self._render()
        elif field_name == "heading":
            prefix = "\n\n" if self.sections_md else ""
            self.sections_md += f"{prefix}### {value}\n"
            self._render()
        elif field_name == "content":
            self.sections_md += "\n"
            self._render()

    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if field_name == "content":
            self.sections_md += chunk
            self._render()

    def on_array_item_end(
        self,
        path: str,
        field_name: str,
        item: Optional[Dict[str, Any]] = None,
    ) -> None:
        if field_name != "references" or not item:
            return
        line = f"- {format_reference_line(item)}"
        path = path or ""
        normalized = path.lstrip("/")
        if normalized.startswith("sections"):
            section_path = path.split("/references", 1)[0]
            if section_path not in self.section_reference_paths:
                self.sections_md += "\n\n#### References\n"
                self.section_reference_paths.add(section_path)
            self.sections_md += line + "\n"
        self._render()

LAST_STRUCTURED_OUTPUT_KEY = "qdrant_last_structured_output"


def stream_agent_response(
    agent,
    question: str,
    event_handler,
    message_history,
    placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, List[Any]]:
    handler = StreamlitSearchResultHandler(placeholder)
    parser = StreamingJSONParser(handler)
    previous_text = ""
    final_payload_text = ""
    logfire.log("info", "agent_stream_start", {"question": question})

    async def _run():
        nonlocal previous_text, final_payload_text
        with logfire.span("agent_run_stream", question=question):
            async with agent.run_stream(
                question,
                event_stream_handler=event_handler,
                message_history=message_history,
            ) as result:
                async for item, _ in result.stream_responses(debounce_by=0.01):
                    for part in getattr(item, "parts", []):
                        tool_name = getattr(part, "tool_name", None)
                        if tool_name != "final_result":
                            continue

                        raw_payload = getattr(part, "args", None)
                        if raw_payload is None:
                            raw_payload = getattr(part, "content", None)

                        current_text = _payload_to_json_text(raw_payload)
                        if not isinstance(current_text, str):
                            continue

                        delta = current_text[len(previous_text) :]
                        parser.parse_incremental(delta)
                        previous_text = current_text
                        final_payload_text = current_text
                try:
                    new_messages = result.new_messages()
                except Exception:
                    new_messages = []
        return new_messages

    new_messages = asyncio.run(_run())
    structured_output: Optional[SearchResultResponse] = None
    final_rendered = handler.rendered_text
    if final_payload_text:
        try:
            structured_output = SearchResultResponse.model_validate_json(final_payload_text)
        except Exception as exc_json:
            logfire.log(
                "error",
                "structured_output_parse_error",
                {
                    "question": question,
                    "error": str(exc_json),
                    "payload": final_payload_text,
                    "method": "model_validate_json",
                },
            )
            try:
                structured_output = SearchResultResponse.model_validate(
                    json.loads(final_payload_text)
                )
            except Exception as exc_model:
                logfire.log(
                    "error",
                    "structured_output_parse_error",
                    {
                        "question": question,
                        "error": str(exc_model),
                        "payload": final_payload_text,
                        "method": "model_validate",
                    },
                )
                structured_output = None
    st.session_state[LAST_STRUCTURED_OUTPUT_KEY] = structured_output

    if structured_output:
        final_rendered = structured_output.format_response()
        handler.rendered_text = final_rendered
        placeholder.markdown(final_rendered)
    elif not final_rendered:
        logfire.log("warning", "agent_stream_missing_payload", {"question": question})
        raise ValueError("No final result payload received.")

    logfire.log(
        "info",
        "agent_stream_complete",
        {"question": question, "structured": bool(structured_output)},
    )
    return handler.rendered_text, new_messages


def make_tool_logger(
    lines: List[str], placeholder: st.delta_generator.DeltaGenerator
):
    tool_labels = {
        "search_embeddings": "Searching Qdrant...",
        "search_web": "Searching the web...",
        "get_page_content": "Fetching content...",
    }

    async def _logger(ctx, event):
        async def _walk(ev):
            if hasattr(ev, "__aiter__"):
                async for nested in ev:
                    await _walk(nested)
                return

            if isinstance(ev, FunctionToolCallEvent):
                tool_name = ev.part.tool_name
                if tool_name == "final_result":
                    return
                args = parse_args(getattr(ev.part, "args", None))

                label = tool_labels.get(tool_name)
                description = f"- `{tool_name}`"
                if label:
                    description += f" · {label}"
                query = args.get("query")
                if query:
                    description += f" · **query:** {query}"
                url = args.get("url")
                if url:
                    description += f" · **url:** {url}"

                lines.append(description)
                placeholder.markdown("\n".join(lines))
                logfire.log(
                    "debug",
                    "tool_call",
                    {"tool_name": tool_name, "args": args},
                )

        await _walk(event)

    return _logger


def build_summary_messages(question: str, summary: str) -> List[Any]:
    """Create compact user/assistant messages from the latest exchange."""
    request = ModelRequest.user_text_prompt(question.strip())
    response = ModelResponse(parts=[TextPart(summary.strip())])
    return [request, response]


SUMMARY_SECTION_LIMIT = 3
SUMMARY_CHARS_PER_SECTION = 240
SUMMARY_FALLBACK_CHARS = 600


def build_abbreviated_summary(
    output: Optional[SearchResultResponse], fallback_text: str
) -> str:
    """Create a compact summary for message history to reduce token usage."""
    if output:
        lines: List[str] = []
        description = (output.description or "").strip()
        if description:
            lines.append(description)
        for section in output.sections[:SUMMARY_SECTION_LIMIT]:
            snippet = (section.content or "").strip()
            if not snippet:
                continue
            snippet = " ".join(snippet.split())
            if len(snippet) > SUMMARY_CHARS_PER_SECTION:
                cutoff = snippet[:SUMMARY_CHARS_PER_SECTION].rsplit(" ", 1)[0]
                snippet = cutoff + "..."
            heading = (section.heading or "Section").strip()
            lines.append(f"{heading}: {snippet}")
        if lines:
            return "\n".join(lines)

    fallback = " ".join(fallback_text.split())
    if len(fallback) > SUMMARY_FALLBACK_CHARS:
        fallback = fallback[:SUMMARY_FALLBACK_CHARS].rsplit(" ", 1)[0] + "..."
    return fallback


HISTORY_WINDOW = 3
TOKEN_ENCODING = os.getenv("TOKEN_GUARD_ENCODING", "cl100k_base")
MODEL_CONTEXT_LIMIT = int(os.getenv("MODEL_CONTEXT_LIMIT", "128000"))
TOKEN_CAP_RATIO = 0.9


st.set_page_config(page_title="Habit Builder AI Agent", page_icon="🌱", layout="wide")
st.title("🌱 Habit Builder AI Agent")
# st.markdown("#### *Turning Research into Daily Action*")
st.markdown("Building lasting habits is challenging especially when you don't understand the why behind them. The Habit Builder AI Agent uses programmatic and automated tools to help you turn complex information into simple habits that you can immediately implement, all supported by the latest research.")
st.markdown("Built with Faster Whisper for transcription, OpenAI GPT models as the engine, Pydantic/PydanticAI for structured agent logic, Qdrant for vector search, and Logfire for end-to-end observability. Curious about how I built this? [Check out my repo.](https://github.com/vtdinh13/habit-builder-ai-agent)")



with st.sidebar:
    st.header("Settings")
    st.caption(
        "Life happens, even to agents. Try reloading then ask another question.")
    if st.button("Reload Agent"):
        load_agent.clear()
        st.success("Agent cache cleared.")

    st.caption("Keep your workspace clean by clearing the current conversation.")
    clear_chat = st.button("Clear Conversation")

    st.caption("Agent making endless searches? Stop the agent.")
    if st.button("Stop loading", type="primary"):
        st.warning("Stopping response...")
        st.stop()
    st.divider()
    st.markdown("**Disclaimer:**")
    st.markdown("This application is an independent creation and is neither endorsed by nor affiliated with the Huberman Lab podcast. *For educational purposes only.*")
    st.markdown(
    "⭐️ **Support the project**: [Star or fork the repository on GitHub](https://github.com/vtdinh13/habit-builder-ai-agent)."
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if clear_chat:
    st.session_state["chat_history"] = []
    st.session_state["message_history"] = []
    st.rerun()


def gather_recent_texts(window: int) -> List[str]:
    texts: List[str] = []
    history = st.session_state.get("chat_history", [])
    recent = history[-window:] if window else history
    for exchange in recent:
        question = (exchange.get("question") or "").strip()
        answer = (exchange.get("answer") or "").strip()
        if question:
            texts.append(question)
        if answer:
            texts.append(answer)
    return texts


for exchange in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.markdown(exchange["question"])
    with st.chat_message("assistant"):
        tool_activity = exchange.get("tool_activity") or []
        if tool_activity:
            st.markdown("**Tool activity**")
            st.markdown("\n".join(tool_activity))
        st.markdown(exchange["answer"])


user_question = st.chat_input(
    "Ask a question about sleep, motivation, neuroscience, or performance."
)

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    logfire.log("info", "user_question_received", {"question": user_question})

    agent = load_agent()

    guard = TokenBudget(
        encoding_name=TOKEN_ENCODING,
        max_context_tokens=MODEL_CONTEXT_LIMIT,
        cap_ratio=TOKEN_CAP_RATIO,
    )
    agent_instructions = getattr(agent, "instructions", None)
    base_texts = gather_recent_texts(HISTORY_WINDOW) + [user_question.strip()]
    if isinstance(agent_instructions, str) and agent_instructions.strip():
        base_texts.append(agent_instructions.strip())
    try:
        guard.initialize(base_texts)
        reserve_tokens = max(1000, int(MODEL_CONTEXT_LIMIT * 0.05))
        guard.consume_tokens(reserve_tokens, label="system_overhead")
    except TokenBudgetExceeded as exc:
        logfire.log(
            "warning",
            "token_guard_block_start",
            {
                "question": user_question,
                "attempted_tokens": exc.attempted,
                "cap": exc.cap,
            },
        )
        st.warning(
            "I'm close to the model's context limit. Please clear the conversation or ask a shorter question."
        )
        st.stop()
    with st.chat_message("assistant"):
        tool_container = st.container()
        with tool_container:
            st.markdown("**Tool activity**")
            tool_placeholder = st.empty()
        tool_lines: List[str] = []

        tool_logger = make_tool_logger(tool_lines, tool_placeholder)
        answer_placeholder = st.empty()

        history = st.session_state["message_history"]
        recent_history = history[-HISTORY_WINDOW:] if HISTORY_WINDOW else history

        with st.spinner("Generating answer..."):
            try:
                with activate_guard(guard):
                    answer_md, new_messages = stream_agent_response(
                        agent,
                        user_question.strip(),
                        tool_logger,
                        recent_history,
                        answer_placeholder,
                    )
            except TokenBudgetExceeded as exc:
                tool_placeholder.empty()
                logfire.log(
                    "warning",
                    "token_guard_triggered",
                    {
                        "question": user_question,
                        "attempted_tokens": exc.attempted,
                        "cap": exc.cap,
                        "label": exc.label,
                    },
                )
                st.warning(
                    "I stopped the tools because we're at the model's context limit. Please narrow your question or clear the chat."
                )
            except Exception as exc:
                tool_placeholder.empty()
                logfire.log(
                    "error",
                    "agent_stream_error",
                    {"question": user_question, "error": str(exc)},
                )
                st.error(f"Something went wrong: {exc}")
            else:
                answer_placeholder.markdown(answer_md)
                logfire.log(
                    "info",
                    "streamlit_agent_response",
                    {
                        "question": user_question,
                        "answer": answer_md,
                        "tool_activity": list(tool_lines),
                    },
                )
                st.session_state["chat_history"].append(
                    {
                        "question": user_question,
                        "answer": answer_md,
                        "tool_activity": list(tool_lines),
                    }
                )
                try:
                    structured_output = st.session_state.get(LAST_STRUCTURED_OUTPUT_KEY)
                    summary_text = build_abbreviated_summary(
                        structured_output, answer_md
                    )
                    compact_messages = build_summary_messages(
                        user_question, summary_text
                    )
                    st.session_state["message_history"].extend(compact_messages)
                except Exception:
                    st.warning(
                        "Unable to store the latest conversation context for follow-up questions."
                    )

if not st.session_state["chat_history"]:
    st.info("Start by asking about sleep, motivation, neuroscience, or performance.")
