import json
import requests
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode


def search_papers(query: str, max_results: int = 10):
    """
    Search arXiv for papers relevant to a research query.
    
    Returns JSON containing paper title, authors, publication date, abstract,
    arXiv URL, PDF URL, and categories.
    """
    
    query = query.strip()
    max_results = max(1, min(max_results, 10))  # Limit max results to 10
    
    sort_by = "submittedDate" if any(
            keyword in query.lower() for keyword in ["recent", "latest", "new"]
        ) else "relevance"
    
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    
    url = f"https://export.arxiv.org/api/query?{urlencode(params)}"
    
    # response = requests.get(
    #     url,
    #     timeout=20,
    #     headers={"User-Agent": "explainable-agentic-rag/0.1"},
    # )
    # response.raise_for_status()
    headers = {
        "User-Agent": "explainable-agentic-rag/0.1 contact: neeraj1909@gmail.com"
    }
    
    last_error = None
    
    for attempt in range(5):
        try:
            response = requests.get(
                url, 
                timeout=(5, 60), 
                headers=headers
            )
        except requests.exceptions.Timeout as exc:
            last_error = str(exc)
            wait_seconds = 5 * (attempt + 1)
            print(f"arXiv rate timed out. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
            continue
        except requests.exceptions.RequestException as exc:
            return json.dumps(
            {
                "query": query,
                "source": "arXiv",
                "error": "arXiv rate limit exceeded. Try again later.",
                "results": [],
            },
            indent=2,
        )
            
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after else 10 * (attempt + 1)
            print(f"arXiv rate limited request. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
            continue
        
        if response.status_code in (500, 502, 503, 504):
            wait_seconds = 5 * (attempt + 1)
            print(f"arXiv server error {response.status_code}. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)
            continue
            
        response.raise_for_status()
        break
    else:
        return json.dumps(
            {
                "query": query,
                "source": "arXiv",
                "error": f"arXiv unavailable after retries. Last error: {last_error}",
                "results": [],
            },
            indent=2,
        )
            
    root = ET.fromstring(response.text)
    
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    
    papers = []
    
    for entry in root.findall("atom:entry", ns):
        title = " ".join(entry.findtext("atom:title", "", ns).split())
        abstract = " ".join(entry.findtext("atom:summary", "", ns).split())
        arxiv_url = entry.findtext("atom:id", "", ns)
        
        authors = [
            author.findtext("atom:name", "", ns)
            for author in entry.findall("atom:author", ns)
        ]
        
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
            
        categories = [
            category.attrib.get("term")
            for category in entry.findall("atom:category", ns)
        ]
        
        papers.append(
            {
                "title": title,
                "authors": authors,
                "published": entry.findtext("atom:published", "", ns),
                "abstract": abstract,
                "arxiv_url": arxiv_url,
                "pdf_url": pdf_url,
                "categories": categories,
            }
        )

    return json.dumps(
        {
            "query": query,
            "source": "arXiv",
            "results": papers,
        },
        indent=2,
    )
    
if __name__ == "__main__":
    query = "machine learning for healthcare"
    print(search_papers(query, max_results=5))
    