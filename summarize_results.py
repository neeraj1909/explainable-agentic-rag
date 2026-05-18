import argparse
import json
import re
from collections import Counter
from datetime import datetime 
from typing import Any, Dict, List, Optional

from search_papers import search_papers

def summarize_search_results(search_result: Dict[str, Any]) -> str:
    """
    Summarize paper search results from arXiv or similar sources.
    
    Input:
        search_result: dict containing query, source, retrieval_method, results
        
    Output:
        Markdown-formatted summary string
    """
    
    query = search_result.get("query", "Unknown query")
    source = search_result.get("source", "Unknown source")
    retrieval_method = search_result.get("retrieval_method", "Unknown method")
    papers = search_result.get("results", [])
    
    if not papers:
        return f"No papers found for query: {query}"
    
    categories = Counter()
    years = []
    
    for paper in papers:
        categories.update(paper.get("categories", []))
        
        published = paper.get("published")
        if published:
            try:
                # years.append(datetime.fromisoformat(published.replace("Z", "+00:00")).year)
                years.append(extract_year(published)) 
            except ValueError:
                pass
            
    summary_lines = []
    
    summary_lines.append(f"# Search Summary")
    summary_lines.append("")
    summary_lines.append(f"**Query:** {query}")
    summary_lines.append(f"**Source:** {source}")
    summary_lines.append(f"**Retrieval Method:** {retrieval_method}")
    summary_lines.append(f"**Total Papers Found:** {len(papers)}")
    
    if years:
        summary_lines.append(f"**Publication Year Range:** {min(years)}-{max(years)}")
        
    if categories:
        top_categories = ", ".join(
            f"{cat} ({count})" for cat, count in categories.most_common(5)
        )
        summary_lines.append(f"**Top Categories:** {top_categories}")

    summary_lines.append("")
    summary_lines.append("## Overall Takeaway")
    #add takaway here
    summary_lines.append("## Papers")
    summary_lines.append("")
    
    for index, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        published = paper.get("published", "")
        year = extract_year(published)
        abstract = paper.get("abstract", "")
        arxiv_url = paper.get("arxiv_url", "")
        
        short_summary = summarize_abstract(abstract)
        
        summary_lines.append(f"### {index}. {title}")
        summary_lines.append("")
        summary_lines.append(f"- **Year:** {year or 'Unknown'}")
        summary_lines.append(f"- **Authors:** {format_authors(authors)}")
        summary_lines.append(f"- **Summary:** {short_summary}")
        summary_lines.append(f"- **URL:** {arxiv_url}")
        summary_lines.append("")
        
    return "\n".join(summary_lines)


def extract_year(published: Optional[str]) -> Optional[int]:
    if not published:
        return None
    
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).year
    except ValueError:
        return None
    
    
def format_authors(authors: List[str], max_authors: int = 3) -> str:
    if not authors:
        return "Unknown"
    
    if len(authors) <= max_authors:
        return ", ".join(authors)
    else:
        return ", ".join(authors[:max_authors]) + f", et al."
    
    
def summarize_abstract(abstract: str, max_sentences: int = 2) -> str:
    """
    Simple extractive summary:
    take the first 1-2 sentences from the abstract
    
    Later, you can replace this with an LLm-based summarizer.
    """
    
    if not abstract:
        return "No abstract available"
    
    sentences = split_sentences(abstract)
    selected = sentences[:max_sentences]
    
    return " ".join(selected)


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.
    For production use, consider nltk, spacy, or an LLM.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]  


def main():
    parser = argparse.ArgumentParser(description="Search and summarize arXiv papers.")
    parser.add_argument("query", nargs="+", help="User search query")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--summary", action="store_true", help="Print summarized output")
    
    args = parser.parse_args()
    
    query = " ".join(args.query)
    raw_result = search_papers(query, max_results=args.max_results)
    if args.summary:
        payload = json.loads(raw_result)
        print(summarize_search_results(payload))
    else:
        print(raw_result)

if __name__ == "__main__":
    main()
