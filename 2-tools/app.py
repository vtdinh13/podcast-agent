import asyncio
import json
from typing import Any

import streamlit as st

from pydantic_ai.messages import FunctionToolCallEvent

from jaxn import JSONParserHandler, StreamingJSONParser

import logfire
from search_agent import create_search_agent

logfire.configure()
logfire.instrument_pydantic_ai()

DEFAULT_MODEL = "openai:gpt-4o-mini"


@st.cache_resource(show_spinner=False)
def load_agent(model_name: str):
    """
    Build and cache the search agent so the sentence transformer/ES client are reused
    across Streamlit reruns.
    """
    agent = create_search_agent(model=model_name)
    return agent


class StreamlitSearchResultHandler(JSONParserHandler):
    """Stream chunks into a Streamlit placeholder."""

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator):
        super().__init__()
        self.placeholder = placeholder
        self.buffer: list[str] = []
        self.rendered_text: str = ""

    def _render(self) -> None:
        text = "".join(self.buffer).strip()
        if not text:
            return
        self.placeholder.markdown(text)
        self.rendered_text = text

    def on_field_start(self, path: str, field_name: str) -> None:
        if field_name == "references" and "sections" not in path:
            return
        if field_name == "references":
            header_level = path.count("/") + 1
            self.buffer.append(f"\n\n{'#' * header_level} References\n")
            self._render()

    def on_field_end(
        self, path: str, field_name: str, value: str, parsed_value: Any = None
    ) -> None:
        if field_name == "heading":
            self.buffer.append(f"\n\n## {value}\n\n")
            self._render()
        elif field_name == "content":
            self.buffer.append("\n")
            self._render()

    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if field_name == "content":
            self.buffer.append(chunk)
            self._render()

    def on_array_item_end(
        self, path: str, field_name: str, item: dict[str, Any] | None = None
    ) -> None:
        if field_name == "references" and "sections" not in path:
            return
        if field_name == "references" and item:
            episode = item.get("episode_name", "")
            window = f"{item.get('start_time', '')}-{item.get('end_time', '')}"
            line = f"- {episode} ({window})\n"
            self.buffer.append(line)
            self._render()


def stream_agent_response(
    agent,
    question: str,
    event_handler,
    message_history,
    placeholder: st.delta_generator.DeltaGenerator,
) -> tuple[str, list[Any]]:
    handler = StreamlitSearchResultHandler(placeholder)
    parser = StreamingJSONParser(handler)
    previous_text = ""

    async def _run():
        nonlocal previous_text
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
                    current_text = part.args
                    delta = current_text[len(previous_text) :]
                    parser.parse_incremental(delta)
                    previous_text = current_text
            try:
                new_messages = result.new_messages()
            except Exception:
                new_messages = []
        return new_messages

    new_messages = asyncio.run(_run())
    if not handler.rendered_text:
        raise ValueError("No final result payload received.")
    return handler.rendered_text, new_messages


st.set_page_config(page_title="Podcast Agent", page_icon="🧠", layout="wide")
st.title(" 🧠 Podcast Agent")
st.markdown(
    "A conversational interface grounded in podcast episodes from the Huberman Lab, powered by `gpt-4o-mini` and PydanticAI. Data ingestion consists of audio downloads via RSS, transcription with Faster-Whisper, embedding generation with Sentence Transformers, and chunk indexing in Elasticsearch. Over 350 episodes were indexed, ranging from December 21, 2021, to November 17, 2025."
)
st.markdown(
    "The agent embeds user queries, retrieves relevant segments from the vector index, and streams time-stamped responses. All interactions are logged via Logfire."
)





with st.sidebar:
    st.markdown("")
    st.markdown("")
    st.header("Configuration")
    st.caption(
        "Life happens, even to agents. Try reloading then ask another question. "
       
    )
    # model_name = st.text_input("Model", value=DEFAULT_MODEL, help="pydantic-ai model id")
    rerun_button = st.button("Reload Agent", help="Clear cached agent for the chosen model")
    if rerun_button:
        load_agent.clear()
        st.success("Agent cache cleared. Next run will reinitialize the model.")
    st.caption("Keep your workspace clean by clearing the current conversation.")
    clear_chat = st.button("Clear Conversation")

    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown(
        "**Disclaimer:**")
    st.markdown("1. This project is actively evolving. Please help by engaging with the agent, as your questions will directly improve their responses."
    )
    st.markdown("2. This application is an independent creation and is neither endorsed by nor affiliated with the Huberman Lab podcast. *For educational purposes only.*")
    st.markdown("")
    st.markdown(
    "⭐️ **Support the project**: [Star or fork the repository on GitHub](https://github.com/vtdinh13/huberman-agent)."
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if clear_chat:
    st.session_state["chat_history"] = []
    st.session_state["message_history"] = []
    st.rerun()

for exchange in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.markdown(exchange["question"])
    with st.chat_message("assistant"):
        tool_activity = exchange.get("tool_activity") or []
        if tool_activity:
            st.markdown("**Tool activity**")
            st.markdown("\n".join(tool_activity))
        st.markdown(exchange["answer"])

user_question = st.chat_input("Ask about sleep, fitness, motivation, neuroscience, general health, etc.")

model_name = 'openai:gpt-4o-mini'
if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    agent = load_agent(model_name)

    with st.chat_message("assistant"):
        tool_container = st.container()
        tool_container.markdown("**Tool activity**")
        tool_activity_placeholder = tool_container.empty()
        tool_lines: list[str] = []

        async def streamlit_tool_logger(ctx, event):
            async def _walk(ev):
                if hasattr(ev, "__aiter__"):
                    async for sub in ev:
                        await _walk(sub)
                    return
                if isinstance(ev, FunctionToolCallEvent):
                    tool_name = ev.part.tool_name

                    if tool_name == "embed_query":
                        description = f"- `{tool_name}` *embedding*"
                    else:
                        description = f"- `{tool_name}` *searching*"

                    args = getattr(ev.part, "args", None)
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    if tool_name == "vector_search" or tool_name == "embed_query":
                        query = args.get("query")
                        if query:
                            description += f" *query: {query}*"
                    tool_lines.append(description)
                    tool_activity_placeholder.markdown("\n".join(tool_lines))
            await _walk(event)

        answer_placeholder = st.empty()
        with st.spinner("Generating answer..."):
            try:
                # Stream the final_result tool output so users can see content
                # as it arrives while still leveraging structured parsing.
                answer_md, new_messages = stream_agent_response(
                    agent,
                    user_question.strip(),
                    streamlit_tool_logger,
                    st.session_state["message_history"],
                    answer_placeholder,
                )
            except Exception as exc:
                tool_activity_placeholder.empty()
                st.error(f"Something went wrong: {exc}")
            else:
                answer_placeholder.markdown(answer_md)
                logfire.log(
                    "streamlit_agent_response",
                    {
                        "user_question": user_question,
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
                    st.session_state["message_history"].extend(new_messages)
                except Exception:
                    st.warning("Unable to store conversation history for follow-up questions.")

if not st.session_state["chat_history"]:
    st.info("Start the conversation using the chat box below.")
