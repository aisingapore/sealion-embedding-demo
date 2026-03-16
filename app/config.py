from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding model
    embedding_model: str = Field(
        default="aisingapore/SEA-LION-Embedding-300M", alias="EMBEDDING_MODEL"
    )

    # LLM settings
    openai_base_url: str = Field(
        default="http://localhost:11434/v1", alias="OPENAI_BASE_URL"
    )
    openai_api_key: str = Field(default="ollama", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="llama3", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")

    # ChromaDB settings
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="sea_lion_docs", alias="CHROMA_COLLECTION")

    # App settings
    top_k: int = Field(default=5, alias="TOP_K")
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"


settings = Settings()
