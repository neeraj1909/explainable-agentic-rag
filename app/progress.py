from langgraph.config import get_stream_writer


def emit_progress(event: str, message: str, **data):
    try:
        writer = get_stream_writer()
        writer({
            "event": event,
            "message": message,
            "data": data,
        })
    except Exception as e:
        # Normal invoke/tests: no active stream writer.
        pass
