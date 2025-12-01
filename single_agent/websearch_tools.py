import requests
import os
import random
from typing import Optional, List
import random 


preferred_sites = [
    "brainfacts",
    "nimh",
    "nih",
    "alleninstitute",
    "mit",
    "stanford",
    "acsm",
    "nsca",
    "acefitness",
    "exerciseismedicine",
    "bjsm",
    "apa",
    "motivationscience",
    "berkeley",
    "mayoclinic",
    "clevelandclinic",
    "harvard",
    "hopkinsmedicine",
    "cdc",
    "mpg",
    "yale",
    "scientificamerican",
    "psychologytoday",
    "nature",
    "science",
]

def web_search(query: str, preferred_sites: List[str]):
    """
    Query Brave Search and keep only results that match trusted domains.

    Args:
        query: Free-form question or keyword string sent to Brave Search.
        preferred_sites: Domain substrings used to filter the Brave results

    Returns:
        Up to five filtered URLs, or None if the Brave request fails.
    """

    url = f"https://api.search.brave.com/res/v1/web/search?q={query}"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.getenv("BRAVE_API_KEY")
    }

    try:
        response = requests.get(url, headers=headers)
        results = response.json().get("web", {}).get("results", [])
        urls_all = [item.get("url") for item in results if item.get("url")]  
        urls_filtered = [u for u in urls_all if any(dom in u for dom in preferred_sites)]

        if len(urls_filtered) > 5:
            urls_filtered = random.sample(urls_filtered, 5)
        return urls_filtered

    except (requests.exceptions.RequestException, UnicodeDecodeError) as e:
        print(f" Error fetching content for {query}: {e}")
        # logging.exception(f" Error fetching content for {query}")
        return None



def get_page_content(url: str) -> Optional[str]:
    """
    Retrieve article content through the Jina reader proxy and decode it into UTF-8 text.

    Args:
        url: The original page URL to fetch.

    Returns:
        Page contents as a string or None if the fetch or decode fails.
    """

    reader_url_prefix = "https://r.jina.ai/"
    reader_url = reader_url_prefix + url

    try:
        response = requests.get(reader_url, timeout=45)
        response.raise_for_status()  # raises for 4xx/5xx HTTP errors
        return response.content.decode("utf-8")
    except (requests.exceptions.RequestException, UnicodeDecodeError) as e:
        # Optional: log or print the error for debugging
        print(f"Error fetching content from {url}: {e}")
        return None
