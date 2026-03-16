import logging

import gradio as gr
import numpy as np

from app import embedder

logger = logging.getLogger(__name__)


def compute_similarity(sentence_a: str, sentence_b: str):
    if not sentence_a.strip() or not sentence_b.strip():
        return 0.0, "—", "—", "Please enter both sentences."

    try:
        embeddings = embedder.encode([sentence_a, sentence_b], prompt_name="STS")
        emb_a = np.array(embeddings[0])
        emb_b = np.array(embeddings[1])

        # Embeddings are already L2-normalized (normalize_embeddings=True)
        cosine_sim = float(np.dot(emb_a, emb_b))
        cosine_sim = max(0.0, min(1.0, cosine_sim))

        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))

        label = (
            "Very similar"
            if cosine_sim >= 0.95
            else (
                "Similar"
                if cosine_sim >= 0.75
                else "Somewhat similar" if cosine_sim >= 0.60 else "Dissimilar"
            )
        )

        return cosine_sim, f"{norm_a:.4f}", f"{norm_b:.4f}", label
    except Exception:
        logger.error("Similarity computation failed", exc_info=True)
        return 0.0, "—", "—", "An error occurred. Please try again."


def build_tab() -> gr.Tab:
    with gr.Tab("Cross-Lingual Similarity") as tab:
        gr.Markdown(
            "## Cross-Lingual Similarity Explorer\n"
            "Compare two sentences in any language and compute their cosine similarity using the SEA-LION embedding model."
        )
        with gr.Row():
            sentence_a = gr.Textbox(
                label="Sentence A",
                placeholder="e.g. The cat sat on the mat.",
                lines=3,
            )
            sentence_b = gr.Textbox(
                label="Sentence B",
                placeholder="e.g. Kucing itu duduk di atas tikar.",
                lines=3,
            )

        with gr.Row():
            compare_btn = gr.Button("Compare", variant="primary", scale=0)
            reset_btn = gr.Button("Reset", size="sm", scale=0, variant="secondary")

        with gr.Row():
            similarity_score = gr.Slider(
                label="Cosine Similarity",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                interactive=False,
            )
            similarity_label = gr.Textbox(label="Interpretation", interactive=False)

        with gr.Row():
            norm_a_out = gr.Textbox(label="Embedding Norm (A)", interactive=False)
            norm_b_out = gr.Textbox(label="Embedding Norm (B)", interactive=False)

        compare_btn.click(
            fn=compute_similarity,
            inputs=[sentence_a, sentence_b],
            outputs=[similarity_score, norm_a_out, norm_b_out, similarity_label],
        )
        reset_btn.click(
            fn=lambda: ("", "", 0.0, "", "", ""),
            inputs=[],
            outputs=[
                sentence_a,
                sentence_b,
                similarity_score,
                norm_a_out,
                norm_b_out,
                similarity_label,
            ],
        )

    return tab
