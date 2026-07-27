"""Compatibility constants sourced from the central application settings."""

from app.config import get_settings


_settings = get_settings()

DOCS_DIR = _settings.docs_dir
INDEX_DIR = _settings.index_dir
CHUNK_SIZE = _settings.chunk_size
CHUNK_OVERLAP = _settings.chunk_overlap
TOP_K = _settings.top_k
FETCH_K_MULTIPLIER = _settings.fetch_k_multiplier
