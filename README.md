# SEA-LION Embedding Demo

A Gradio web app showcasing multilingual semantic search and retrieval-augmented generation (RAG) using [SEA-LION](https://sea-lion.ai/) embedding models. Documents are indexed into a ChromaDB vector store and queries can be answered by an LLM grounded in the retrieved context.

**Architecture:** Gradio UI → SentenceTransformers (SEA-LION embeddings) → ChromaDB → OpenAI-compatible LLM

**Languages supported:** English, Malay, Indonesian, Thai, Vietnamese, Filipino, and other Southeast Asian languages.

---

## Requirements

- **Docker & Docker Compose** — for the recommended setup
- **Python 3.11+ and [uv](https://github.com/astral-sh/uv)** — for local development without Docker
- An **OpenAI-compatible LLM endpoint** — see [LLM Setup](#llm-setup) below

---

## Embedding Model

The app uses a SEA-LION embedding model from HuggingFace. By default it uses `aisingapore/SEA-LION-ModernBERT-Embedding-300M`, which is downloaded automatically on first run into the HuggingFace cache (`~/.cache/huggingface/`).

Browse the full collection of SEA-LION embedding models here: <https://huggingface.co/aisingapore/collections>

To pre-download a model manually:

```bash
pip install huggingface_hub
hf download aisingapore/SEA-LION-ModernBERT-Embedding-300M
```

> **Note:** If the model is saved to a non-default location (i.e. not `~/.cache/huggingface/`), set the `HF_CACHE_PATH` environment variable to point to that directory so the app can find it.
>
> **Docker users:** If you have pre-downloaded the model on the host, mount the cache directory into the container and set `HF_CACHE_PATH` to the mounted path in your `.env`. Without this, Docker will re-download the model on every fresh container start.

---

## LLM Setup

A SEA-LION LLM is recommended for multilingual quality for South-East Asian languages. The app connects to any OpenAI-compatible API endpoint, configured via `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

### Option A — Ollama (easiest local setup)

Install [Ollama](https://ollama.com), then pull a SEA-LION model:

```bash
ollama pull aisingapore/Qwen-SEA-LION-v4-32B-IT
```

Full list of SEA-LION models on Ollama: <https://ollama.com/aisingapore?sort=newest>

Set in `.env`:
```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=aisingapore/Qwen-SEA-LION-v4-32B-IT
```

### Option B — SEA-LION API

Sign up and obtain an API key at <https://playground.sea-lion.ai/key-manager>, then set `OPENAI_BASE_URL` to the SEA-LION API endpoint <https://api.sea-lion.ai/v1> and `OPENAI_API_KEY` to your key.

### Option C — Other OpenAI-compatible endpoints

vLLM, Amazon Bedrock Access Gateway, Google Vertex AI, and others are supported. See <https://docs.sea-lion.ai/guides/inferencing> for the full list of deployment options.

---

## Quickstart — Docker Compose

```bash
cp .env.example .env        # edit LLM_MODEL, OPENAI_BASE_URL, OPENAI_API_KEY
docker compose up --build -d
```

Open <http://localhost:7860> once the Gradio app container is running.

ChromaDB and the Gradio app run as separate services. Document index data is persisted in a named Docker volume (`chroma_data`).

---

## Local Dev Setup (uv)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Set CHROMA_HOST=localhost in .env

# 5. Start ChromaDB (separate terminal)
docker run -p 8000:8000 chromadb/chroma

# 6. Run the app
python -m app.main
```

Open <http://localhost:7860>.

---

## Environment Variables

Copy `.env.example` to `.env` and edit as needed. All variables are optional — defaults are shown below.

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `aisingapore/SEA-LION-ModernBERT-Embedding-300M` | HuggingFace model ID for embeddings |
| `HF_CACHE_PATH` | *(unset — uses `~/.cache/huggingface/`)* | Override HuggingFace cache directory; required when the model is pre-downloaded to a non-default path and when running in Docker with a mounted model cache |
| `OPENAI_BASE_URL` | `http://host.docker.internal:11434/v1` | LLM API base URL, use `http://localhost:11434/v1` if running local dev setup |
| `OPENAI_API_KEY` | `ollama` | LLM API key (`ollama` for local Ollama) |
| `LLM_MODEL` | `aisingapore/Qwen-SEA-LION-v4-32B-IT` | Model name passed to the LLM API |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature for RAG answers |
| `CHROMA_HOST` | `localhost` | ChromaDB host (`chromadb` when using Docker Compose) |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `CHROMA_COLLECTION` | `sea_lion_docs` | ChromaDB collection name |
| `TOP_K` | `5` | Number of document chunks to retrieve per query |
| `CHUNK_SIZE` | `500` | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Character overlap between consecutive chunks |

> **Note:** When using Docker Compose, `CHROMA_HOST` is automatically set to `chromadb` (the service name) via `docker-compose.yml`. You do not need to set it manually in that case.

---

## App Tabs

### Semantic Search

Enter a query in any supported language to retrieve the most semantically similar document chunks from the index. Results are displayed in a ranked table with similarity scores and source document names.

### Cross-Lingual Similarity

Compare two sentences — in the same or different languages — and compute their cosine similarity using the SEA-LION embedding model. The result includes a similarity score, embedding norms, and an interpretation label (Dissimilar → Somewhat similar → Similar → Very similar).

### RAG Q&A

Ask a question in any language. The system:
1. Detects the language of the question
2. Retrieves the most relevant document chunks from the index
3. Passes them as context to the LLM to generate a grounded answer
4. Returns the answer along with cited source documents (expandable Sources panel)

### Document Management

- View all indexed documents with chunk counts, last-modified timestamps, and source folders
- Remove a specific document from the index by name
- Trigger a full re-index of the `documents/` folder to pick up new or changed files

---

## Adding Documents

Place files in the `documents/` folder. On startup (or after clicking **Re-index** in the Manage tab), the app will extract text, chunk it, embed it, and store it in ChromaDB. Files that have not changed since the last run are skipped.

**Supported formats:** `.txt` `.md` `.rst` `.yaml` `.yml` `.json` `.csv` `.xml` `.html` `.htm` `.pdf` `.docx`

The `sample_data/` folder contains pre-loaded example documents and is mounted read-only in Docker.

---

## Project Structure

```
embedding-demo/
├── app/
│   ├── config.py        — Pydantic settings (env vars → Python)
│   ├── embedder.py      — SentenceTransformer singleton and encode()
│   ├── vectorstore.py   — ChromaDB CRUD and search operations
│   ├── indexer.py       — File reading, chunking, and indexing pipeline
│   ├── rag.py           — LLM client, language detection, RAG answer()
│   ├── main.py          — Gradio app entry point
│   └── tabs/            — One module per UI tab
│       ├── search.py
│       ├── similarity.py
│       ├── rag_qa.py
│       └── manage.py
├── documents/           — User-uploaded files (persisted via Docker volume)
├── sample_data/         — Pre-loaded read-only example documents
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
