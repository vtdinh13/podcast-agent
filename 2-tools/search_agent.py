import search_tools
from dataclasses import dataclass

from pydantic import BaseModel
import json
from pydantic_ai import Agent



search_instructions = """
You are Huberman Lab Search, a search assistant with access only to the Huberman Lab podcast archive (topics include but not limited to sleep, motivation, neuroscience, fitness, general health).

Given a user question, your goal is to retrieve the most relevant podcast passages and synthesize a response supported by those passages in natural language. 


SEARCH STRATEGY
- Rewrite the user question in three distinct ways (e.g., different phrasing, key terms, related subquestions).
- You have access to the vector search tool. For each rewrite, call the `vector_search` tool once. Record the references (episode name, start/end).
- Merge the retrieved chunks, synthesize an answer based on the retrieved chunks in natural language, and cite every statement with its reference metadata. Make sure you include the rephrased question in your response.
- If no relevant chunks are found after all rewrites, state that explicitly and offer general guidance.


RULES
- Use only information returned from the vector search tool; never invent facts.
- Always provide the rephrased question, numbered sections, and a reference list with timestamps.
- Write your answer clearly and accurately.
- Always include a reference section listing with time stamps and guest name(s) from the podcast episodes.

CONTEXT:
---
{chunk}
---

""".strip()


class Reference(BaseModel):
    title: str
    episode_name: str
    start_time: str
    end_time:str

class Section(BaseModel):
    heading: str
    content: str
    references: list[Reference]

class SearchResultResponse(BaseModel):
    rephrased_question: str
    sections: list[Section]
    references: list[Reference]

    def format_response(self) -> str:
        output = "## Your Question\n\n"
        output += f"{self.rephrased_question}\n\n"

        for section in self.sections:
            output += f"## {section.heading}\n\n"
            output += f"{section.content}\n\n"
            if section.references:
                output += "### References\n"
                for ref in section.references:
                    output += f" {ref.title}, {ref.episode_name}, {ref.start_time}, {ref.end_time}\n"
            output += "\n"

        return output.strip()


# @dataclass
# class AgentConfig:
#     index_name: str = "huberman"
#     model: str = "openai:gpt-4o-mini"


def create_search_agent(model:str) -> Agent:
    
    search_tool = search_tools.SearchTools()

    def vector_search(query: str):
        """Vector search constrained to the configured index."""
        results = search_tool.vector_search(
            query=query
        )
        return json.dumps(results, indent=2)

    search_agent = Agent(
        name='search_agent',
        instructions=search_instructions,
        tools=[vector_search],
        model=model,
        output_type=SearchResultResponse,
        # history_processors=[force_answer_after_4_searches],
    )

    return search_agent






















# from pydantic_ai.messages import FunctionToolCallEvent

# class NamedCallback:

#     def __init__(self, agent):
#         self.agent_name = agent.name

#     async def print_function_calls(self, ctx, event):
#         # Detect nested streams
#         if hasattr(event, "__aiter__"):
#             async for sub in event:
#                 await self.print_function_calls(ctx, sub)
#             return

#         if isinstance(event, FunctionToolCallEvent):
#             tool_name = event.part.tool_name
#             args = event.part.args
#             print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

#     async def __call__(self, ctx, event):
#         return await self.print_function_calls(ctx, event)

# from pydantic_ai.messages import ModelMessage, UserPromptPart
# def force_answer_after_4_searches(messages: list[ModelMessage]) -> list[ModelMessage]: 
#     num_tool_calls = 0

#     for m in messages:
#         for p in m.parts:
#             if p.part_kind == 'tool-call' and p.tool_name == 'search':
#                 num_tool_calls = num_tool_calls + 1

#     if num_tool_calls >= 4:
#         print('forcing output')
#         last_message = messages[-1]
#         finish_prompt = 'System message: The maximal number of searches has exceeded 4. Proceed to finishing the writeup'
#         finish_prompt_part = UserPromptPart(finish_prompt)
#         last_message.parts.append(finish_prompt_part)

#     return messages