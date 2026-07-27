from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from app.config import AppSettings, get_settings

_instrumented = False


def setup_phoenix_tracing(settings: AppSettings | None = None):
    global _instrumented
    if _instrumented:
        return

    resolved = settings or get_settings()
    if not resolved.phoenix_enabled:
        return

    tracer_provider = register(
        project_name=resolved.phoenix_project_name,
        endpoint=str(resolved.phoenix_collector_endpoint),
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _instrumented = True
