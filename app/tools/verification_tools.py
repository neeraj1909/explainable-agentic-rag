import json 
import re


STOPWORDS = {
    "the", "a", "an", "and", "or", "off", "to", "in", "for", "on", "with",
    "is", "are", "was", "were", "by", "as", "that", "this", "it", "from"
}


def _sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]
    
    
def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text)
        if token.lower() not in STOPWORDS and len(token) > 2
    }
    

def calculate_faithfulness_stub(
    answer: str,
    evidence: str,
    threshold: float = 0.35,
) -> str:
    """
    Estimate whether an answer is faithful to provided evidence.
    
    Args:
        answer: The generated claim, summary or answer to verify.
        evidence: Retrieved abstracts, paper snippets, or source text.
        threshold: Minimum token-coverage score for a sentence to count as supported.
    
    Returns:
        JSON string with faithfulness_score, unsupported_claims, and method details.
    """
    
    answer_sentences = _sentences(answer)
    evidence_tokens = _tokens(evidence)
    
    if not answer_sentences:
        return json.dumps({
            "faithfulness_score": 0.0,
            "unsupported_claims": [],
            "verdict": "no_answer_provided",
            "method": "token_overlap_stub"
        }, indent=2)
        
    if not evidence_tokens:
        return json.dumps({
            "faithfulness_score": 0.0,
            "unsupported_claims": answer_sentences,
            "verdict": "no_evidence_provided",
            "method": "token_overlap_stub"
        }, indent=2)
        
    sentence_scores = []
    unsupported_claims = []
    
    for sentence in answer_sentences:
        sentence_tokens = _tokens(sentence)
        
        if not sentence_tokens:
            continue
        
        overlap_score = len(sentence_tokens & evidence_tokens) / len(sentence_tokens)
        sentence_scores.append(overlap_score)
        
        if overlap_score < threshold:
            unsupported_claims.append(sentence)
        
    faithfulness_score = (
        sum(sentence_scores) / len(sentence_scores)
        if sentence_scores 
        else 0.0
    )
    
    verdict = (
        "faithful"
        if faithfulness_score >= threshold and not unsupported_claims
        else "partially_faithful"
        if faithfulness_score >= threshold
        else "low_faithfulness"
    )
    
    return json.dumps({
        "faithfulness_score": round(faithfulness_score, 3),
        "unsupported_claims": unsupported_claims,
        "verdict": verdict,
        "method": "token_overlap_stub",
        "threshold": threshold, 
    }, indent=2)
 