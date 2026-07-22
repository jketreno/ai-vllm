# backend/src/api/tools/web_search.py
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --------- Choose the client ----------------------------------
# Option A: DuckDuckGo (no API key needed)
from duckduckgo_search import DDGS

# Option B: SerpAPI (uncomment if you use SerpAPI)
# from serpapi import GoogleSearch
# SERPAPI_KEY = os.getenv("SERPAPI_KEY")
# ----------------------------------------------------------------

router = APIRouter(prefix="/tools/web_search", tags=["tools"])


class SearchResult(BaseModel):
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Brief snippet / description")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(..., description="Top N results")
    source: str = Field(..., description="Which backend powered the results")


def _duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Thin wrapper around duckduckgo-search."""
    with DDGS() as ddg:
        raw = ddg.text(query, safesearch="Moderate", max_results=max_results)
        # ddg.text returns a dict per result with keys: title, url, body (snippet)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]


# To use SerpAPI instead, uncomment the imports above and add a
# _serpapi_search(query, max_results) function following the DuckDuckGo
# wrapper's shape, then switch the call site in run_search() below.


@router.post("/", response_model=SearchResponse)
async def run_search(payload: Dict[str, Any]):
    """
    Entry point called by Open‑WebUI when the LLM invokes the `web_search`
    tool.  Expected JSON payload:

    ```json
    {
        "arguments": {
            "query": "your search string",
            "max_results": 5          # optional
        }
    }
    ```

    Returns a `SearchResponse` that will be rendered as markdown in the UI.
    """
    args = payload.get("arguments", {})
    query: str = args.get("query", "").strip()
    if not query:
        raise HTTPException(
            status_code=400, detail="`query` must be a non‑empty string"
        )

    max_results: int = int(args.get("max_results", 5))
    max_results = max(1, min(max_results, 10))  # safety clamp

    # ----- Choose which backend to call --------------------------------
    try:
        # For DuckDuckGo
        raw_results = _duckduckgo_search(query, max_results=max_results)
        source = "DuckDuckGo"
        # For SerpAPI comment the line above and uncomment below:
        # raw_results = _serpapi_search(query, max_results=max_results)
        # source = "SerpAPI"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Convert to pydantic models
    results = [SearchResult(**r) for r in raw_results]

    return SearchResponse(query=query, results=results, source=source)
