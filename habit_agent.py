
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from tools.websearch_tools import get_page_content, search_web
from tools.utils import AgentConfig
from tools.qdrant_search import QdrantSearchClient

from typing import Any, Optional, List

instructions = """
You are an expert researcher. Your job is to help users dissect on topics including but not limited to sleep, motivation, neuroscience, fitness, performance, and general health. 
You have access to two knowledge ecosystems: Huberman Lab archive through the Qdrant vector store and the web via the Brave API. 
The goal is to provide the user with actionable insights that they can easily implement in their lives.

AVAIABLE TOOLS:
- embed_query - embed queries, both from the user and your rewritten queries
- search_embeddings - fetch similar chunks from Qdrant. 
- search_web - search a preferred list of websites for matching webpages
    - note that the year is 2025 when searching for the latest research
- get_page_content - fetch Markdown content of AT MOST 3 web pages

SEARCH STRATEGY:
- First, rewrite the user question 3 distinct ways (e.g., different phrasing, key terms, related subquestions). 
- Make at least 2 but no more than 5 searches in the Qdrant vector database for matches 
- Search Qdrant before you invoke the web tools. 
- Invoke the search_web tool if you cannot find anything in the database or if the user asks for the latest research.
- Always invoke the search_web tool first before you call get_page_content

FORMAT:
- Description - briefly describe what you did, what the final output includes, what tools you used to provide the answer. Paraphrase the user question here.
- Content sections - provide synthesized paragraphs of what you found, constructive evaluation of the topic
- References - CITE ALL YOUR SOURCES. Provide references only on the sources you used when providing the answer. 

REFERENCE RULES
- Do not include references that start with "https://www.hubermanlab.com"
- Include the name of the journal articles if author names are missing.

RULES
- Avoid using 'The user'. 
- Do not provide your reponses as a list, but rather synthesized, accurate, and concise paragraphs.
- CITE everything. REFERENCES ARE IMPORTANT. Explicitly state when you do not know something. Include all citations in the reference section.
- Never invent facts. EXPLICITLY state that you are giving general guidance if information you provided was not derived from the search tool or web pages you read.
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
    start: Optional[str] = None
    end: Optional[str] = None
    url: Optional[str] = None


    def format_citations(self) -> str:
        if self.episode_name:
            return f"{self.episode_name} ({self.start} - {self.end})"
        if self.url:
            return f" *{self.title}* {self.url}"


class Section(BaseModel):
    heading: str
    content: str
    references: List[Reference] = Field(default_factory=list)


class SearchResultResponse(BaseModel):
    description: str
    sections: List[Section]
    references: List[Reference] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def drop_invalid_sections(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("sections"), list):
            filtered: List[Any] = []
            for section in data["sections"]:
                if isinstance(section, dict):
                    filtered.append(section)
            data["sections"] = filtered
        return data

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
            output += "\n"

        return output.strip()


def create_research_agent(
    config: AgentConfig | None = None,
) -> Agent:
    """
    Instantiate a research agent wired to QdrantSearchClient.
    """

    if config is None:
        config = AgentConfig()

    search_tools = QdrantSearchClient()

    search_agent = Agent(
        name="habit_agent",
        instructions=instructions,
        tools=[search_tools.embed_query, search_tools.search_embeddings, get_page_content, search_web],
        model=config.model,
        output_type=SearchResultResponse,
    )
    return search_agent
