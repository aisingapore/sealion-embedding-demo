import logging

from openai import OpenAI

from app import embedder, vectorstore
from app.config import settings

logger = logging.getLogger(__name__)

_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    return _llm_client


def detect_language(text: str) -> str:
    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": f"What language is this text written in? Reply with only the language name.\n\nText: {text[:500]}",
                }
            ],
            max_tokens=20,
            temperature=0,
        )
        if not resp.choices or not resp.choices[0].message.content:
            logger.warning(
                "Language detection returned empty response, instructing LLM to reply in the same language"
            )
            return "the same language as the query"
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(
            "Language detection failed, instructing LLM to reply in the same language: %s",
            e,
            exc_info=True,
        )
        return "the same language as the query"


def answer(question: str) -> tuple[str, list[dict]]:
    detected_lang = detect_language(question)
    logger.info(f"Detected language: {detected_lang}")

    try:
        query_embedding = embedder.encode([question], prompt_name="Retrieval")[0]
    except RuntimeError:
        logger.error("Failed to embed question", exc_info=True)
        return (
            "The embedding model was unable to encode the query. Please try again.",
            [],
        )

    try:
        hits = vectorstore.search(query_embedding, top_k=settings.top_k)
    except RuntimeError:
        logger.error("Vector store search failed in answer()", exc_info=True)
        return (
            "The knowledge base encountered an error when searching. Please try again.",
            [],
        )

    if not hits:
        return "No relevant documents found in the knowledge base.", []

    context_parts = []
    for i, hit in enumerate(hits, 1):
        source = hit["metadata"]["source"]
        text = hit["text"]
        context_parts.append(f"[{i}] Source: {source}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        f"You are a helpful assistant. Respond in {detected_lang}. "
        "Use only the provided context to answer the question. "
        "If the context does not contain enough information, say so. "
        "Cite sources by referencing the source document name."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.llm_temperature,
        )
        if not resp.choices or not resp.choices[0].message.content:
            logger.error("LLM returned empty response")
            answer_text = (
                "An error occurred while generating the answer. Please try again."
            )
        else:
            answer_text = resp.choices[0].message.content.strip()
    except Exception:
        logger.error("LLM call failed", exc_info=True)
        answer_text = "An error occurred while generating the answer. Please try again."

    return answer_text, hits
