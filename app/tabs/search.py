import html
import logging

import gradio as gr

from app import embedder, vectorstore
from app.config import settings

logger = logging.getLogger(__name__)

_TABLE_STYLE = (
    "width:100%;table-layout:fixed;border-collapse:collapse;"
)
_CELL_STYLE = "padding:0.4em 0.6em;border:1px solid var(--border-color-primary,#333);vertical-align:top;"
_TH_STYLE = _CELL_STYLE + "white-space:nowrap;"


def _hits_to_html(hits: list[dict]) -> str:
    body_rows = []
    for i, hit in enumerate(hits, 1):
        text = " ".join((hit["text"] or "").split())
        if len(text) > 300:
            text = text[:300] + "..."
        body_rows.append(
            "<tr>"
            f'<td style="{_CELL_STYLE}text-align:center;">{i}</td>'
            f'<td style="{_CELL_STYLE}text-align:center;">{hit["score"]:.4f}</td>'
            f'<td style="{_CELL_STYLE}word-break:break-word;">'
            f'{html.escape(hit["metadata"]["source"])}</td>'
            f'<td style="{_CELL_STYLE}word-break:break-word;">{html.escape(text)}</td>'
            "</tr>"
        )
    return (
        f'<table style="{_TABLE_STYLE}">'
        "<colgroup>"
        '<col style="width:5%" />'
        '<col style="width:10%" />'
        '<col style="width:25%" />'
        '<col style="width:60%" />'
        "</colgroup>"
        "<thead><tr>"
        f'<th style="{_TH_STYLE}">Rank</th>'
        f'<th style="{_TH_STYLE}">Score</th>'
        f'<th style="{_TH_STYLE}">Source Document</th>'
        f'<th style="{_TH_STYLE}">Text Excerpt</th>'
        "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def do_search(query: str) -> str:
    if not query.strip():
        return "<p><em>Enter a query and click <strong>Search</strong>.</em></p>"
    try:
        query_embedding = embedder.encode([query], prompt_name="Retrieval")[0]
        hits = vectorstore.search(query_embedding, top_k=settings.top_k)
        if not hits:
            return "<p><em>No results found.</em></p>"
        return _hits_to_html(hits)
    except Exception:
        logger.error("Search failed", exc_info=True)
        raise gr.Error("Search failed. Please try again.")


def build_tab() -> gr.Tab:
    with gr.Tab("Semantic Search") as tab:
        gr.Markdown(
            "## Multilingual Semantic Search\n"
            "Enter a query in any language and retrieve the most semantically similar document chunks."
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g. nasi lemak, Songkran festival, teknologi digital...",
                lines=2,
                scale=4,
            )
            with gr.Column(scale=0, min_width=100):
                search_btn = gr.Button("Search", variant="primary")
                reset_btn = gr.Button("Reset", size="sm", variant="secondary")

        results_output = gr.HTML()

        search_btn.click(fn=do_search, inputs=query_input, outputs=results_output)
        query_input.submit(fn=do_search, inputs=query_input, outputs=results_output)
        reset_btn.click(
            fn=lambda: (
                "",
                "<p><em>Enter a query and click <strong>Search</strong>.</em></p>",
            ),
            inputs=[],
            outputs=[query_input, results_output],
        )

    return tab
