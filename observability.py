import os

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor 

_instrumented = False


def setup_phoenix_tracing():
    global _instrumented
    if _instrumented:
        return

    tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "explainable-agentic-rag"),
        endpoint=os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://10.20.30.1:16006/v1/traces",
        ),
    )
    
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _instrumented = True
