import logging

import gradio as gr

from app import embedder, vectorstore
from app.config import settings

logger = logging.getLogger(__name__)


def do_search(query: str) -> list[list]:
    if not query.strip():
        return []
    try:
        query_embedding = embedder.encode([query], prompt_name="Retrieval")[0]
        hits = vectorstore.search(query_embedding, top_k=settings.top_k)
        rows = []
        for i, hit in enumerate(hits, 1):
            rows.append(
                [
                    i,
                    f"{hit['score']:.4f}",
                    hit["metadata"]["source"],
                    hit["text"][:300] + ("..." if len(hit["text"]) > 300 else ""),
                ]
            )
        return rows
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

        results_table = gr.Dataframe(
            headers=["Rank", "Score", "Source Document", "Text Excerpt"],
            datatype=["number", "str", "str", "str"],
            label="Search Results",
            interactive=False,
            wrap=True,
        )

        search_btn.click(fn=do_search, inputs=query_input, outputs=results_table)
        query_input.submit(fn=do_search, inputs=query_input, outputs=results_table)
        reset_btn.click(
            fn=lambda: ("", []),
            inputs=[],
            outputs=[query_input, results_table],
        )

    return tab
