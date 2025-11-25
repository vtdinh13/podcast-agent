import json
from datetime import datetime
import re
from typing import List, Optional, Union, Set

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent

from search_tools import prepare_search_tools

search_instructions = """
You are an assistant that specializes in finding relevant passages from the Huberman Lab podcast archive (topics include but not limited to sleep, motivation, neuroscience, fitness, general health). Compile and embed all queries and do one vector search with all keywords.

SEARCH STRATEGY
- Rewrite the user question between 2 to 5 distinct ways (e.g., different phrasing, key terms, related subquestions). 
- Merge the retrieved chunks, synthesize an answer based on the retrieved chunks in natural language, and cite every statement with its reference metadata. Make sure you include the rephrased question in your response.
- Paraphrase the user's query clearly and include this in your final response.
- If no relevant chunks are found after all rewrites, state that explicitly and offer general guidance.

TOOLS YOU CAN USE
- embed_query() - Embed rewritten queries
- vector_search() - Fetch relevant chunks

RULES
- Use ALL queries, both the user's and the ones you rewrote, to call the vector search tool 2 to 4 times. STOP AT FOUR, but it could be less.
- Use only information returned from the vector search tool; never invent facts. EXPLICITLY state that you are giving general guidance if information you provided was not derived from the search tool.
- Always provide the paraphrased question, numbered sections, and reference for each statement you make.
- Write your answer clearly and accurately.
- Always include a reference section listing with time stamps and guest name(s) from the podcast episodes.

CONTEXT:
---
{chunk}
---

""".strip()

class Reference(BaseModel):
    episode_name: str
    start_time: str
    end_time:str

class Section(BaseModel):
    heading: str
    content: str
    references: list[Reference]

    def formatted_references(self) -> list[str]:
        lines = []
        for idx, ref in enumerate(self.references, start=1):
            lines.append(
                f"{idx}. {ref.episode_name} ({ref.start_time}-{ref.end_time})"
            )
        return lines

class SearchResultResponse(BaseModel):
    rephrased_question: str
    sections: list[Section]
    references: list[Reference]

    def format_response(self) -> str:
        output = "### Your Question\n\n"
        output += f"{self.rephrased_question}\n\n"

        for section in self.sections:
            output += f"### {section.heading}\n\n"
            output += f"{section.content}\n\n"
            if section.references:
                output += "#### References\n"
                for ref_line in section.formatted_references():
                    output += f" {ref_line}\n"
            output += "\n"

        return output.strip()


def create_search_agent(model:str) -> Agent:

    prepared_tools = prepare_search_tools()

    search_agent = Agent(
        name='search_agent',
        instructions=search_instructions,
        tools=[prepared_tools.embedding, prepared_tools.search],
        model=model,
        output_type=SearchResultResponse,
        # history_processors=[force_answer_after_4_searches],
    )

    return search_agent

class NamedCallback:

    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)
