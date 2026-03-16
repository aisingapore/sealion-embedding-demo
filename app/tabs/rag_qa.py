import logging

import gradio as gr

from app import rag

logger = logging.getLogger(__name__)


def do_rag(question: str):
    if not question.strip():
        return "Please enter a question.", ""

    try:
        answer_text, hits = rag.answer(question)
    except Exception:
        logger.error("RAG pipeline failed", exc_info=True)
        return "An error occurred. Please try again.", ""

    sources_md = ""
    if hits:
        lines = ["### Sources Used\n"]
        for i, hit in enumerate(hits, 1):
            source = hit["metadata"]["source"]
            excerpt = hit["text"][:200].replace("\n", " ")
            score = hit["score"]
            lines.append(f"**[{i}] {source}** (score: {score:.4f})\n> {excerpt}...\n")
        sources_md = "\n".join(lines)
    else:
        sources_md = "_No sources retrieved._"

    return answer_text, sources_md


def build_tab() -> gr.Tab:
    with gr.Tab("RAG Q&A") as tab:
        gr.Markdown(
            "## Retrieval-Augmented Generation Q&A\n"
            "Ask a question in any language. The system retrieves relevant document chunks and uses an LLM to generate an answer."
        )
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What is Songkran? / Apa itu nasi lemak? / เทศกาลสงกรานต์คืออะไร?",
            lines=3,
        )
        with gr.Row():
            ask_btn = gr.Button("Ask", variant="primary", scale=5)
            reset_btn = gr.Button(
                "Reset", size="sm", scale=1, min_width=70, variant="secondary"
            )

        answer_output = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False,
        )

        with gr.Accordion("Sources", open=False):
            sources_output = gr.Markdown()

        ask_btn.click(
            fn=do_rag,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            fn=do_rag,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )
        reset_btn.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[question_input, answer_output, sources_output],
        )

    return tab
