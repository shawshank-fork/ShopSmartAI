import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Astra DB
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RAG_MODEL = "llama-3.1-8b-instant"

    # product API
    PRODUCT_API_BASE = "https://dummyjson.com"
    PRODUCT_CATEGORIES = None

    # Analytics DB
    ANALYTICS_DB_URL = os.getenv("ANALYTICS_DB_URL", "sqlite:///analytics.db")