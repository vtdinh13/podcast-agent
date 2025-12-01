import json
from datetime import datetime
import re

from pydantic import BaseModel
from pydantic_ai import Agent

from dataclasses import dataclass 

from search_tools import prepare_search_tools
from websearch_tools import get_page_content, web_search

from utils import AgentConfig

from typing import Optional

instructions = """
You are a specialized research assistant. Your job is to help users dissect on topics including but not limited to sleep, motivation, neuroscience, fitness, performance, and general health. 
You have access to two knowledge ecosystems: Huberman Lab podcast through the elastic search vector store and the web via the Brave API. 

SEARCH STRATEGY, always in this order:
- First, rewrite the user question 3 distinct ways (e.g., different phrasing, key terms, related subquestions). Compile and embed all queries.
- Then, do ONE vector search with embeddings from all queries. Merge the retrieved chunks, synthesize an answer based on the retrieved chunks in natural language, and cite every statement with its reference metadata. Make sure you include the rephrased question in your response.
- At this point, the user may ask you questions on the information you provided. Explicitly state if you are providing general guidance; cite your sources otherwise.
- Next, continue to do more research if the user asks for further information or current research. Only then should you call the Brave API.

AVAIABLE TOOLS:
- search_embeddings - embed queries, both from the user and your rewritten queries then fetch similar chunks from elastic search. 
    - NEVER INVOKE MORE THAN 3 search_embeddings call at once.
- websearch - search a list of specified websites for matching webpages
    - note that the year is 2025 when searching for the latest research
- get_page_content - fetch Markdown content of AT MOST 5 web pages

FORMAT:
- Description - briefly describe what you did, what the final output includes, what tools you used to provide the answer. Paraphrase the user question here.
- Content sections - this depends on whether you used the vector store or web
    - vector store: provide synthesized paragraphs of what you found, preferably one section where you considered and state all of the viewpoints and provide a constructive evaluation fo the topic.
    - web: ENSURE TO INCLUDE TWO COMPONENTS 
        1. concise but detailed and accurate summaries of the web pages you found. Explain key arguments, evidence, findings, methods, assumptions, strengths and limitations.
        2. synthesis across all articles, identify dissonance across articles, extract core insights and patterns, identify novelty and emerging themes
- References - CITE ALL YOUR SOURCES. Provide references only on the sources you used when providing the answer. 


REFERENCE RULES
- divided into 2 different formats depending on whether you searched vector store or web
    - vector store 
        1. ALWAYS INCLUDE episode name, start and end times.
        2. ALWAYS INCLUDE name of podcast participants or title of podcast in inline citations
    - web
        1. NEVER SEARCH on the huberman website with urls that start with "https://www.hubermanlab.com"; look in the vector store insead.
        2. Include website names when author names are missing in "References".
- DO NOT PROVIDE LINKS INLINE

RULES
- Avoid using 'The user'. 
- ADHERE TO REFERENCE RULES.
- Do not provide your reponses as a list, but rather synthesized, accurate, and concise paragraphs.
- YOU MUST PROVIDE LINKS TO ALL WEB PAGES IN YOUR RESPONSE IN THE REFERENCE SECTION. 
- CITE everything. REFERENCES ARE IMPORTANT. Explicitly state when you do not know something. Include all citations in the reference section.
- Use only information returned from the vector search tool or web pages; never invent facts. EXPLICITLY state that you are giving general guidance if information you provided was not derived from the search tool or web pages you read.
- For each response, rewrite the user's question clearly in the description and ensure that you are answering the question that you rewrote.
- Your reponse must be clear and accurate.


CONTEXT:
---
{chunk}
---

""".strip()

class Reference(BaseModel):
    title: Optional[str] = None
    episode_name: Optional[str] = None
    author: Optional[str] = None
    date_article_is_written: Optional[int] = None
    url: Optional[str] = None
    start_time: Optional[str]=None
    end_time: Optional[str]=None
    def format_citations(self) -> str:
        if self.episode_name:
            return f"{self.episode_name} ({self.start_time}-{self.end_time})"
        if self.url:
            date_article_is_written = self.date_article_is_written or "Unknown Date"
            author = self.author or "Unknown Authors"
            title = self.title or "Unknown Title"
            url = self.url
            return f" {author}. {date_article_is_written}. *{title}* {url}".strip()

class Section(BaseModel):
    heading: str
    content: str
    references: list[Reference]

class SearchResultResponse(BaseModel):
    description: str
    sections: list[Section]

    def format_response(self) -> str:
        output = "### Description\n\n"
        output += f"{self.description}\n\n"

        for section in self.sections:
            output += f"### {section.heading}\n\n"
            output += f"{section.content}\n\n"
            if section.references:
                output += "#### References\n"
                for reference in section.references:
                    output += f" - {reference.format_citations()}\n"
                    # output += f" - {reference}\n"
            output += "\n"

        return output.strip()



def create_search_agent(config: AgentConfig = None) -> Agent:

    if config is None:
        config = AgentConfig()

    prepared_tools = prepare_search_tools()

    search_agent = Agent(
        name='research_agent',
        instructions=instructions,
        tools=[prepared_tools.search, get_page_content, web_search],
        model=config.model,
        output_type=SearchResultResponse,
    )

    return search_agent

