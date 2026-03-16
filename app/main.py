import logging
import sys

import gradio as gr

from app import indexer
from app.config import settings
from app.tabs import manage, rag_qa, search, similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ── Startup: index files ──────────────────────────────────────────────────────
logger.info("Starting indexer...")
try:
    indexer.scan_and_index(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    logger.info("Indexer complete.")
except Exception:
    logger.error(
        "Startup indexing failed; app will launch with existing index", exc_info=True
    )

# ── Build Gradio app ──────────────────────────────────────────────────────────

_model_name = settings.embedding_model
_model_url = f"https://huggingface.co/{_model_name}"

with gr.Blocks(title="SEA-LION Embedding Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SEA-LION Embedding Demo\n"
        f"Powered by [`{_model_name}`]({_model_url}) — "
        "a multilingual embedding model for Southeast Asian languages.\n\n"
        "**Languages supported:** English, Malay, Indonesian, Thai, Vietnamese, Filipino, and more."
    )

    search.build_tab()
    similarity.build_tab()
    rag_qa.build_tab()
    manage.build_tab()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
