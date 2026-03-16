import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model {settings.embedding_model}...")
        model_kwargs = {}

        # Provide template prompt for different tasks for SEA-LION embedding models for better results
        if "SEA-" in settings.embedding_model:
            model_kwargs["prompts"] = {
                "STS": "Instruct: Retrieve semantically similar text.\nQuery: ",
                "Clustering": "Instruct: Classify text into its appropriate category\nQuery: ",
                "Classification": "Instruct: Classify text into its appropriate category\nQuery: ",
                "Retrieval": "Instruct: Given a passage that is guaranteed to contain the answer, retrieve relevant passages that answer the query.\nQuery: ",
                "BitextMining": "Instruct: Retrieve parallel sentences.\nQuery: ",
                "PairClassification": "Instruct: Retrieve semantically similar text.\nQuery: ",
                "Reranking": "Instruct: Retrieve semantically similar text.\nQuery: ",
                "InstructionRetrieval": "Instruct: Given a instruction and a output, retrieve the most relevant output that answer the instruction.\nQuery: ",
                "MultiLabelTextClassification": "Instruct: Classify the given text into its appropriate classes\nQuery: ",
                "QARetrieval": "Instruct: Given a passage that is guaranteed to contain the answer, retrieve relevant passages that answer the query.\nQuery: ",
                "Summarization": "Instruct: Summarize the given text.\nQuery: ",
            }
        try:
            _model = SentenceTransformer(settings.embedding_model, **model_kwargs)
        except Exception:
            logger.error(
                "Failed to load embedding model '%s'",
                settings.embedding_model,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load embedding model '{settings.embedding_model}'."
            )
        logger.info("Model loaded successfully.")
    return _model


def encode(texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
    model = get_model()
    encode_kwargs: dict = dict(show_progress_bar=False, normalize_embeddings=True)

    # Insert prompt name to match the task the embedder is intended to perform.
    # For SEA-LION Embedding models
    if prompt_name is not None and "SEA-" in settings.embedding_model:
        encode_kwargs["prompt_name"] = prompt_name
    try:
        embeddings = model.encode(texts, **encode_kwargs)
        return embeddings.tolist()
    except Exception:
        logger.error("Embedding encode failed", exc_info=True)
        raise RuntimeError("Failed to generate embeddings.")
