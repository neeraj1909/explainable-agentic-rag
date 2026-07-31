from collections.abc import Iterator, Sequence

import pytest

from app.bootstrap import build_rag_service
from app.contracts import (
    ClaimStatus,
    ClaimVerification,
    EvidenceChunk,
    ProgressEvent,
    RAGMode,
    RAGRequest,
    RAGResponse,
    VerificationStatus,
    VerificationSummary,
)
from app.ports.retrieval import RetrievalPort
from app.ports.verification import VerificationPort
from app.services.rag_service import (
    ModeHandler,
    RunContext,
    UnsupportedRAGModeError,
)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> Sequence[EvidenceChunk]:
        self.calls.append((query, top_k))
        return [
            EvidenceChunk(
                document_id="doc-1",
                chunk_id="chunk-1",
                source="docs/example.pdf",
                page=2,
                content="The evidence is grounded in a local document.",
                selected_rank=1,
                reason_selected="Highest ranked fake result.",
            )
        ]


class FakeVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def verify(
        self,
        *,
        answer: str,
        evidence: Sequence[EvidenceChunk],
    ) -> VerificationSummary:
        chunk_ids = [chunk.chunk_id for chunk in evidence]
        self.calls.append((answer, chunk_ids))
        return VerificationSummary(
            status=VerificationStatus.verified,
            verified=True,
            score=0.95,
            method="fake_verifier",
            claims=[
                ClaimVerification(
                    claim_id="claim-1",
                    claim=answer,
                    status=ClaimStatus.supported,
                    evidence_chunk_ids=chunk_ids,
                    score=0.95,
                    reason="Supported by the fake evidence.",
                )
            ],
        )


class FakeMode:
    def __init__(
        self,
        retriever: RetrievalPort,
        verifier: VerificationPort,
    ) -> None:
        self.retriever = retriever
        self.verifier = verifier

    def answer(self, request: RAGRequest, context: RunContext) -> RAGResponse:
        evidence = list(self.retriever.retrieve(request.query, top_k=request.top_k))
        answer = "A grounded answer from the fake mode."
        verification = self.verifier.verify(answer=answer, evidence=evidence)
        return RAGResponse(
            mode=request.mode,
            answer=answer,
            evidence=evidence,
            verification=verification,
        )

    def stream(
        self,
        request: RAGRequest,
        context: RunContext,
    ) -> Iterator[ProgressEvent | RAGResponse]:
        yield ProgressEvent(
            event="retrieval_started",
            message="Retrieving fake evidence.",
        )
        yield self.answer(request, context)


def fixed_run_context(request: RAGRequest) -> RunContext:
    return RunContext(
        run_id="run-test-1",
        trace_id="0123456789abcdef0123456789abcdef",
        corpus_version="corpus-test-1",
        index_version="index-test-1",
    )


def build_fake_service(*, streaming: bool = True):
    retriever = FakeRetriever()
    verifier = FakeVerifier()
    mode = FakeMode(retriever, verifier)
    handler = ModeHandler(
        answer=mode.answer,
        stream=mode.stream if streaming else None,
    )
    service = build_rag_service(
        mode_handlers={RAGMode.two_step: handler},
        run_context_provider=fixed_run_context,
    )
    return service, retriever, verifier


def test_answer_uses_offline_ports_and_adds_run_context() -> None:
    service, retriever, verifier = build_fake_service()
    request = RAGRequest(
        query="What is grounded?",
        mode=RAGMode.two_step,
        top_k=3,
    )

    response = service.answer(request)

    assert isinstance(retriever, RetrievalPort)
    assert isinstance(verifier, VerificationPort)
    assert retriever.calls == [("What is grounded?", 3)]
    assert verifier.calls == [("A grounded answer from the fake mode.", ["chunk-1"])]
    assert response.mode is RAGMode.two_step
    assert response.verification.verified is True
    assert response.run_id == "run-test-1"
    assert response.trace_id == "0123456789abcdef0123456789abcdef"
    assert response.corpus_version == "corpus-test-1"
    assert response.index_version == "index-test-1"


def test_stream_normalizes_progress_and_finishes_with_canonical_response() -> None:
    service, retriever, verifier = build_fake_service()
    request = RAGRequest(
        query="Stream a grounded answer.",
        mode=RAGMode.two_step,
        stream=True,
    )

    items = list(service.stream(request))

    assert len(items) == 3
    started, retrieval, response = items
    assert isinstance(started, ProgressEvent)
    assert started.event == "run_started"
    assert started.sequence == 0
    assert started.mode is RAGMode.two_step
    assert isinstance(retrieval, ProgressEvent)
    assert retrieval.event == "retrieval_started"
    assert retrieval.sequence == 1
    assert retrieval.run_id == "run-test-1"
    assert retrieval.trace_id == "0123456789abcdef0123456789abcdef"
    assert isinstance(response, RAGResponse)
    assert response.run_id == "run-test-1"
    assert retriever.calls == [("Stream a grounded answer.", None)]
    assert len(verifier.calls) == 1


def test_stream_falls_back_to_answer_when_mode_has_no_stream_handler() -> None:
    service, _, _ = build_fake_service(streaming=False)
    request = RAGRequest(query="Fallback", mode=RAGMode.two_step)

    items = list(service.stream(request))

    assert [type(item) for item in items] == [ProgressEvent, RAGResponse]
    assert items[0].event == "run_started"
    assert items[1].answer == "A grounded answer from the fake mode."


def test_unsupported_mode_fails_before_constructing_run_context() -> None:
    context_calls: list[RAGRequest] = []
    service, _, _ = build_fake_service()
    service = build_rag_service(
        mode_handlers=service.mode_handlers,
        run_context_provider=lambda request: (
            context_calls.append(request) or fixed_run_context(request)
        ),
    )

    with pytest.raises(UnsupportedRAGModeError, match="agentic"):
        service.answer(RAGRequest(query="Unsupported", mode=RAGMode.agentic))

    assert context_calls == []
