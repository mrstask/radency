import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Pinecone settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-search")

    # Model settings
    MODEL_NAME: str = "all-MiniLM-L6-v2"

    # API settings
    TOP_K: int = 3


settings = Settings()
