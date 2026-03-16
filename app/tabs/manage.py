import datetime
import logging

import gradio as gr

from app import indexer, vectorstore
from app.config import settings

logger = logging.getLogger(__name__)


def _format_mtime(ts: float) -> str:
    if not ts:
        return "—"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _folder_for(source: str) -> str:
    sample = indexer.SAMPLE_DATA_DIR / source
    if sample.exists():
        return "sample_data/"
    return "documents/"


def list_docs() -> list[list]:
    try:
        docs = vectorstore.list_documents()
        rows = []
        for doc in sorted(docs, key=lambda d: d["source"]):
            rows.append(
                [
                    doc["source"],
                    doc["chunk_count"],
                    _format_mtime(doc.get("last_modified", 0)),
                    _folder_for(doc["source"]),
                ]
            )
        return rows
    except Exception:
        logger.error("Failed to list documents", exc_info=True)
        raise gr.Error("Failed to list documents. Please try again.")


def remove_doc(source: str) -> tuple[str, list[list]]:
    if not source or not source.strip():
        return "Please enter a document name to remove.", list_docs()
    pattern = source.strip()
    try:
        deleted = vectorstore.delete_document(pattern)
    except Exception:
        logger.error("Failed to remove document '%s'", pattern, exc_info=True)
        return "Failed to remove document. Please try again.", []
    if deleted == 0:
        return f"No documents matched '{pattern}'.", list_docs()
    return (
        f"Removed {deleted} chunk(s) matching '{pattern}' from the index.",
        list_docs(),
    )


def reindex(progress=gr.Progress()) -> tuple[str, list[list]]:
    progress(0, desc="Starting re-index...")
    try:
        indexer.scan_and_index(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except Exception:
        logger.error("Re-index failed", exc_info=True)
        return "Re-index failed. Please try again.", []
    progress(1, desc="Done")
    return "Re-index complete.", list_docs()


def build_tab() -> gr.Tab:
    with gr.Tab("Document Management") as tab:
        gr.Markdown(
            "## Document Management\n"
            "View indexed documents, remove them from the vector store, or trigger a re-index of the `documents/` folder."
        )

        refresh_btn = gr.Button("Refresh List")
        docs_table = gr.Dataframe(
            headers=["Document Name", "Chunk Count", "Last Modified", "Source Folder"],
            datatype=["str", "number", "str", "str"],
            label="Indexed Documents",
            interactive=False,
        )

        with gr.Row():
            remove_input = gr.Textbox(
                label="Document Name to Remove",
                placeholder="e.g. sea_food_en.txt or sea_*.txt",
            )
            remove_btn = gr.Button("Remove", variant="stop")

        reindex_btn = gr.Button("Re-index documents/ folder now", variant="secondary")
        status_msg = gr.Textbox(label="Status", interactive=False)

        refresh_btn.click(fn=list_docs, outputs=docs_table)
        remove_btn.click(
            fn=remove_doc, inputs=remove_input, outputs=[status_msg, docs_table]
        )
        reindex_btn.click(fn=reindex, outputs=[status_msg, docs_table])

        # Load docs on tab render
        tab.select(fn=list_docs, outputs=docs_table)

    return tab
