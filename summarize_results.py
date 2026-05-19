import argparse
import json
import re
from collections import Counter
from datetime import datetime 
from typing import Any, Dict, List, Optional

from llm_client import get_llm_client
from search_papers import search_papers


def summarize_claim(search_result: Dict[str, Any], llm_client) -> str:
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
    takeaway = generate_overall_takeaway(query, papers, llm_client)
    summary_lines.append(takeaway)
    summary_lines.append("")
    summary_lines.append("## Papers")
    summary_lines.append("")
    
    for index, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        published = paper.get("published", "")
        year = extract_year(published)
        abstract = paper.get("abstract", "")
        arxiv_url = paper.get("arxiv_url", "")
        
        short_summary = summarize_abstract(abstract, llm_client)
                
        summary_lines.append(f"### {index}. {title}")
        summary_lines.append("")
        summary_lines.append(f"- **Year:** {year or 'Unknown'}")
        summary_lines.append(f"- **Authors:** {format_authors(authors)}")
        summary_lines.append(f"- **Summary:** {short_summary}")
        summary_lines.append(f"- **URL:** {arxiv_url}")
        summary_lines.append("")
        
    return "\n".join(summary_lines)


def call_llm(llm_client, prompt: str) -> str:
    response = llm_client.invoke(prompt)
    return response.content.strip()

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
    

def generate_overall_takeaway(user_query: str, paper_summaries: list[dict], llm_client) -> str:
    """
    Generate an overall takeaway from the search results using an LLM.
    
    Input:
        user_query: The original search query
        paper_summaries: List of dicts with keys 'title', 'abstract', 'categories'
        llm_client: An instance of an LLM client to generate the takeaway
    
    Output:
        A string summarizing the overall insights from the papers in relation to the user query.
    """
    # Construct a prompt for the LLM
    prompt = f"""
        Given the following search query and paper summaries, provide an overall takeaway that captures the main insights and trends related to the query.
        User searched for: {user_query}
        Here are summaries of the retrieved papers: {json.dumps(paper_summaries)}
        
        Write an overall takeaway in 3-5 sentences.
        Mention:
        - Common themes across the papers
        - dominant methodologies or approaches
        - any notable gaps or future directions
        - what the user should read first.
        Do not invent claims beyond the provided summaries. 
    """
    # Call the LLM to generate the takeaway
    return call_llm(llm_client, prompt)


def summarize_abstract(abstract: str, llm_client) -> str:
    """
        Summarize the abstract of a paper in relation to the user's query using an LLM-based summarizer.
    """
    
    # if not abstract:
    #     return "No abstract available"
    
    # sentences = split_sentences(abstract)
    # selected = sentences[:max_sentences]
    
    # return " ".join(selected)
    prompt = f"""
        You are summarizing an abstract of an academic paper for a literature search.
        Abstract: {abstract}
        
        Return a concise summary of the abstract in 2-3 sentences, focusing on the aspects most relevant to the user's query.
        Do not include information that is not present in the abstract.
    """
    # Call the LLM to generate the summary
    response = call_llm(llm_client, prompt)
    return response

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
    llm_client = get_llm_client()
    if args.summary:
        payload = json.loads(raw_result)
        print(summarize_claim(payload, llm_client))
    else:
        print(raw_result)

if __name__ == "__main__":
    main()
