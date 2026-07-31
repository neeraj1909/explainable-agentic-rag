"""Application service for canonical RAG answer and stream use cases."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from app.contracts import ProgressEvent, RAGMode, RAGRequest, RAGResponse


class RAGServiceError(RuntimeError):
    """Base error raised by the application-service boundary."""


class UnsupportedRAGModeError(RAGServiceError):
    """Raised when no handler has been registered for a requested mode."""


class RAGServiceContractError(RAGServiceError):
    """Raised when a mode handler violates the canonical service contract."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """Stable execution identity supplied once for an answer or stream."""

    run_id: str
    trace_id: str | None = None
    corpus_version: str | None = None
    index_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        for field_name in ("trace_id", "corpus_version", "index_version"):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name)),
            )


class RunContextProvider(Protocol):
    """Create one run context without coupling the service to tracing SDKs."""

    def __call__(self, request: RAGRequest) -> RunContext: ...


AnswerHandler = Callable[[RAGRequest, RunContext], RAGResponse]
StreamItem = ProgressEvent | RAGResponse
StreamHandler = Callable[[RAGRequest, RunContext], Iterable[StreamItem]]


@dataclass(frozen=True, slots=True)
class ModeHandler:
    """Injected callables for one RAG mode."""

    answer: AnswerHandler
    stream: StreamHandler | None = None


class RAGService:
    """Dispatch canonical RAG use cases to injected mode implementations."""

    def __init__(
        self,
        *,
        mode_handlers: Mapping[RAGMode, ModeHandler],
        run_context_provider: RunContextProvider,
    ) -> None:
        if not mode_handlers:
            raise ValueError("at least one RAG mode handler is required")
        self._mode_handlers = MappingProxyType(dict(mode_handlers))
        self._run_context_provider = run_context_provider

    @property
    def mode_handlers(self) -> Mapping[RAGMode, ModeHandler]:
        """Return the immutable handler registry for composition and inspection."""

        return self._mode_handlers

    def answer(self, request: RAGRequest) -> RAGResponse:
        """Execute one mode and return its normalized canonical response."""

        handler = self._handler_for(request.mode)
        context = self._run_context_provider(request)
        return self._invoke_answer(handler, request, context)

    def stream(self, request: RAGRequest) -> Iterator[StreamItem]:
        """Yield normalized progress followed by exactly one canonical response."""

        handler = self._handler_for(request.mode)
        context = self._run_context_provider(request)
        sequence = 0

        yield ProgressEvent(
            event="run_started",
            message=f"Started {request.mode.value} RAG run.",
            sequence=sequence,
            mode=request.mode,
            run_id=context.run_id,
            trace_id=context.trace_id,
        )
        sequence += 1

        if handler.stream is None:
            yield self._invoke_answer(handler, request, context)
            return

        response_seen = False
        for item in handler.stream(request, context):
            if response_seen:
                raise RAGServiceContractError(
                    "a mode stream must end after its canonical response"
                )
            if isinstance(item, ProgressEvent):
                yield self._normalize_event(item, request, context, sequence)
                sequence += 1
                continue
            if isinstance(item, RAGResponse):
                response_seen = True
                yield self._normalize_response(item, request, context)
                continue
            raise RAGServiceContractError(
                "a mode stream may yield only ProgressEvent or RAGResponse"
            )

        if not response_seen:
            raise RAGServiceContractError(
                "a mode stream must yield one canonical response"
            )

    def _handler_for(self, mode: RAGMode) -> ModeHandler:
        try:
            return self._mode_handlers[mode]
        except KeyError as error:
            available = ", ".join(
                registered.value for registered in self._mode_handlers
            )
            raise UnsupportedRAGModeError(
                f"RAG mode {mode.value!r} is not configured; available: {available}"
            ) from error

    def _invoke_answer(
        self,
        handler: ModeHandler,
        request: RAGRequest,
        context: RunContext,
    ) -> RAGResponse:
        response = handler.answer(request, context)
        if not isinstance(response, RAGResponse):
            raise RAGServiceContractError(
                "a mode answer handler must return RAGResponse"
            )
        return self._normalize_response(response, request, context)

    @staticmethod
    def _normalize_response(
        response: RAGResponse,
        request: RAGRequest,
        context: RunContext,
    ) -> RAGResponse:
        if response.mode is not request.mode:
            raise RAGServiceContractError(
                "response mode does not match the requested mode"
            )

        payload = response.model_dump(mode="python")
        for field_name in (
            "run_id",
            "trace_id",
            "corpus_version",
            "index_version",
        ):
            payload[field_name] = _merge_context_value(
                field_name=field_name,
                current=payload[field_name],
                expected=getattr(context, field_name),
            )
        return RAGResponse.model_validate(payload)

    @staticmethod
    def _normalize_event(
        event: ProgressEvent,
        request: RAGRequest,
        context: RunContext,
        sequence: int,
    ) -> ProgressEvent:
        if event.mode is not None and event.mode is not request.mode:
            raise RAGServiceContractError(
                "progress-event mode does not match the requested mode"
            )

        payload = event.model_dump(mode="python")
        payload.update(
            sequence=sequence,
            mode=request.mode,
            run_id=_merge_context_value(
                field_name="run_id",
                current=event.run_id,
                expected=context.run_id,
            ),
            trace_id=_merge_context_value(
                field_name="trace_id",
                current=event.trace_id,
                expected=context.trace_id,
            ),
        )
        return ProgressEvent.model_validate(payload)


def _merge_context_value(
    *,
    field_name: str,
    current: str | None,
    expected: str | None,
) -> str | None:
    if expected is None:
        return current
    if current is not None and current != expected:
        raise RAGServiceContractError(
            f"{field_name} does not match the service run context"
        )
    return expected


def _require_text(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    return stripped


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
