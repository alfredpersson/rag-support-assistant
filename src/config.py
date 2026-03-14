import os

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "wix_kb"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Timeout (seconds) for individual LLM calls via pydantic-ai ModelSettings.
LLM_TIMEOUT = 30
