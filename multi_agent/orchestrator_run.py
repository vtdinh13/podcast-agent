import asyncio
import json

import orchest_agent


def serialize(content):
    if isinstance(content, str):
        return content
    if hasattr(content, "model_dump"):
        content = content.model_dump()
    elif hasattr(content, "dict"):
        content = content.dict()
    return json.dumps(content, ensure_ascii=False)

async def main():
    user_input = "alzheimer's and coffee"

    agent, callback = orchestv1.initialize_orchestrator()
    
    previous_text = ""
    final_payload = None
    final_plain_text = ""

    async with agent.run_stream(
        user_input, event_stream_handler=callback
    ) as result:
        async for item, last in result.stream_responses(debounce_by=0.01):
            for part in item.parts:
                part_kind = getattr(part, "part_kind", "")
                if part_kind == "text":
                    text_value = str(
                        getattr(part, "content", "") or getattr(part, "text", "")
                    )
                    if text_value:
                        final_plain_text = text_value

                tool_name = getattr(part, "tool_name", None)
                if tool_name != "final_result":
                    continue

                final_payload = part.args
                current_text = serialize(part.args)
                delta = current_text[len(previous_text) :]
                if delta:
                    print(delta, end="", flush=True)
                previous_text = current_text

    if not previous_text:
        if final_payload is not None:
            print(serialize(final_payload))
        elif final_plain_text:
            print(final_plain_text)
        else:
            print("No final result was returned by the orchestrator.")

if __name__ == "__main__":
    asyncio.run(main())
