from typing import List, Tuple

import numpy as np
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from app.config import settings


class SimilaritySearchSystem:
    def __init__(self):
        self.model = SentenceTransformer(settings.MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        if settings.INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=settings.INDEX_NAME,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )

        self.index = self.pc.Index(settings.INDEX_NAME)

    def preprocess_document(self, text: str) -> List[str]:
        """Split document into sentences."""
        return sent_tokenize(text)

    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences using the transformer model."""
        return self.model.encode(sentences)

    def store_embeddings(self, sentences: List[str], embeddings: np.ndarray):
        """Store embeddings in Pinecone vector database."""
        vectors = [(str(i), embedding.tolist(), {"sentence": sentence})
                   for i, (sentence, embedding) in enumerate(zip(sentences, embeddings))]

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search(self, query: str, top_k: int = settings.TOP_K) -> List[Tuple[str, float]]:
        """Search for most similar sentences."""
        query_embedding = self.model.encode(query).tolist()

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return [(match.metadata["sentence"], float(match.score))
                for match in results.matches]

    def initialize_with_document(self, document_text: str):
        """Initialize the system with document text."""
        sentences = self.preprocess_document(document_text)
        embeddings = self.generate_embeddings(sentences)
        self.store_embeddings(sentences, embeddings)
