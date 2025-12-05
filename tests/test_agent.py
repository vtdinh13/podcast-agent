from habit_agent import SearchResultResponse
import main
from .utils import get_tool_calls

def test_search_references():
    user_prompt = "What is the association between coffee and Alzheimer's?"
    result = main.run_agent_sync(user_prompt=user_prompt)

    response: SearchResultResponse = result.output
    print(response.format_response())

    assert len(response.references) >=1, f"Expecting at least one reference, got {len(response.references)}"
    # assert all(section.references for section in response.sections), "Expecting at least one reference in each section."

def test_get_page_content_no_more_than_3():
    user_prompt = "What is the latest research on coffee and Alzheimer's?"
    result = main.run_agent_sync(user_prompt=user_prompt)

    tool_calls = get_tool_calls(result)
    get_page_content_calls = [t for t in tool_calls if t.name == "get_page_content"]

    assert len(get_page_content_calls) <= 3, f"Expecting no more than 3 tool alls, got {len(get_page_content_calls)}"

