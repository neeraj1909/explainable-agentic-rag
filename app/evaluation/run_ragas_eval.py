from __future__ import annotations

import json
import os
from pathlib import Path 

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ragas import EvaluationDataset, evaluate 
from ragas.llms import LangchainLLMWrapper 
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._answer_relevance import ResponseRelevancy

from app.config import get_embedding_client
from app.rag.config import TOP_K
from app.rag.prompts import rag_prompt
from app.rag.retriever import build_attributed_retriever
from app.rag.two_step_rag import format_context 

load_dotenv()


# ----------------------------------------------------------------------
# 1. Load 10-20 manually curated questions + reference answers.
#    References are written from docs/ source material.
# ----------------------------------------------------------------------
EVAL_DATASET_PATH = Path(__file__).with_name("eval_dataset.jsonl")


def load_eval_set(path: Path = EVAL_DATASET_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as dataset_file:
        return [
            json.loads(line)
            for line in dataset_file
            if line.strip()
        ]


EVAL_SET = load_eval_set()


def validate_eval_set() -> None:
    if not (10 <= len(EVAL_SET) <= 20):
        raise ValueError("EVAL_SET should contain 10-20 questions.")
    
    for index, row in enumerate(EVAL_SET, start=1):
        if row["reference"].startswith("TODO"):
            raise ValueError(
                f"Question {index} still has a TODO reference. "
                "Write the ground-truth answer before running Ragas."
            )
            

def build_non_streaming_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["LITELLM_MODEL"],
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ.get("LITELLM_API_BASE"),
        temperature=0,
        streaming=False, 
    )
    
    
def run_two_step_rag_for_eval(question: str, k: int = TOP_K) -> dict:
    """
    Mirrors the existing two-step RAG flow, but also preserves 
    raw retrieved_contexts because Ragas requires them.
    """
    
    retriever = build_attributed_retriever(k=k)
    llm =build_non_streaming_chat_llm()
    chain = rag_prompt | llm | StrOutputParser()
    
    docs = retriever.invoke(question)
    context_text = format_context(docs)
    
    answer = chain.invoke(
        {
            "question": question,
            "context": context_text,
        }
    )
    
    return {
        "response": answer,
        "retrieved_contexts": [doc.page_content for doc in docs],
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "retriever_score": doc.metadata.get("retriever_score"),
                "reranker_score": doc.metadata.get("reranker_score"),
                "selected_rank": doc.metadata.get("selected_rank"),
            }
            for doc in docs
        ],
    }
    
    
def main() -> None:
    validate_eval_set()
    
    rows = []
    
    for item in EVAL_SET:
        question = item["user_input"]
        reference = item["reference"]
        
        rag_result = run_two_step_rag_for_eval(question)
        
        rows.append(
            {
                "user_input": question,
                "response": rag_result["response"],
                "retrieved_contexts": rag_result["retrieved_contexts"],
                "reference": reference,
            }
        )
        
    dataset = EvaluationDataset.from_list(rows)
    
    evaluator_llm = LangchainLLMWrapper(build_non_streaming_chat_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_client())
    
    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        FactualCorrectness(),
        ResponseRelevancy(strictness=1),
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )
    
    print(result)
    
    df = result.to_pandas()
    
    print("\nPer-question results:")
    print(df.to_string())
    
    output_path = Path("evaluation/ragas_eval_results.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved CSV outside codebase: {output_path}")
    
    
if __name__ == "__main__":
    main()
