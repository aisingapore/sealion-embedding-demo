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


def _docs_to_markdown(docs: list[dict]) -> str:
    if not docs:
        return "_No indexed documents._"
    lines = [
        "| Document Name | Chunks | Last Modified | Source Folder |",
        "| --- | ---: | --- | --- |",
    ]
    for doc in sorted(docs, key=lambda d: d["source"]):
        lines.append(
            "| {source} | {chunk_count} | {last_modified} | {folder} |".format(
                source=doc["source"],
                chunk_count=doc["chunk_count"],
                last_modified=_format_mtime(doc.get("last_modified", 0)),
                folder=_folder_for(doc["source"]),
            )
        )
    return "\n".join(lines)


def list_docs() -> str:
    try:
        docs = vectorstore.list_documents()
        return _docs_to_markdown(docs)
    except Exception:
        logger.error("Failed to list documents", exc_info=True)
        raise gr.Error("Failed to list documents. Please try again.")


def remove_doc(source: str) -> tuple[str, str]:
    if not source or not source.strip():
        return "Please enter a document name to remove.", list_docs()
    pattern = source.strip()
    try:
        deleted = vectorstore.delete_document(pattern)
    except Exception:
        logger.error("Failed to remove document '%s'", pattern, exc_info=True)
        return "Failed to remove document. Please try again.", "_Could not refresh list._"
    if deleted == 0:
        return f"No documents matched '{pattern}'.", list_docs()
    return (
        f"Removed {deleted} chunk(s) matching '{pattern}' from the index.",
        list_docs(),
    )


def reindex(progress=gr.Progress()) -> tuple[str, str]:
    progress(0, desc="Starting re-index...")
    try:
        indexer.scan_and_index(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except Exception:
        logger.error("Re-index failed", exc_info=True)
        return "Re-index failed. Please try again.", "_Could not refresh list._"
    progress(1, desc="Done")
    return "Re-index complete.", list_docs()


def build_tab() -> gr.Tab:
    with gr.Tab("Document Management") as tab:
        gr.Markdown(
            "## Document Management\n"
            "View indexed documents, remove them from the vector store, or trigger a re-index of the `documents/` folder."
        )

        refresh_btn = gr.Button("Refresh List")
        docs_table = gr.Markdown(
            value="_Click **Refresh List** to load indexed documents._",
            label="Indexed Documents",
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
