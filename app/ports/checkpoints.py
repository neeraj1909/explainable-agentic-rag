"""Framework-neutral checkpoint boundary for resumable RAG workflows."""

from typing import Protocol, TypeVar, runtime_checkable


CheckpointState = TypeVar("CheckpointState")


@runtime_checkable
class CheckpointPort(Protocol[CheckpointState]):
    """Load and save workflow state under a stable thread identifier."""

    def load(self, thread_id: str) -> CheckpointState | None: ...

    def save(self, thread_id: str, state: CheckpointState) -> None: ...
