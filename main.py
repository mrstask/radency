import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import pinecone
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download NLTK data for sentence tokenization
nltk.download('punkt')


class SearchQuery(BaseModel):
    query: str


class SearchResult(BaseModel):
    sentence: str
    similarity_score: float


app = FastAPI(title="Document Similarity Search API")


class SimilaritySearchSystem:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENV')
        )

        # Get or create index
        self.index_name = "document-search"
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.model.get_sentence_embedding_dimension(),
                metric="cosine"
            )
        self.index = pinecone.Index(self.index_name)

    def preprocess_document(self, text: str) -> List[str]:
        """Split document into sentences."""
        return sent_tokenize(text)

    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences using the transformer model."""
        return self.model.encode(sentences)

    def store_embeddings(self, sentences: List[str], embeddings: np.ndarray):
        """Store embeddings in Pinecone vector database."""
        # Create (id, vector, metadata) tuples for upsert
        vectors = [(str(i), embedding.tolist(), {"sentence": sentence})
                   for i, (sentence, embedding) in enumerate(zip(sentences, embeddings))]

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for most similar sentences."""
        # Generate embedding for query
        query_embedding = self.model.encode(query).tolist()

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract sentences and scores
        matches = [(match.metadata["sentence"], float(match.score))
                   for match in results.matches]

        return matches


# Initialize the system
search_system = SimilaritySearchSystem()


@app.post("/search", response_model=List[SearchResult])
async def search_endpoint(query: SearchQuery) -> List[SearchResult]:
    """Search endpoint that returns top 3 similar sentences."""
    results = search_system.search(query.query)
    return [SearchResult(sentence=sent, similarity_score=score)
            for sent, score in results]


def initialize_system(document_text: str):
    """Initialize the system with document text."""
    # Preprocess document
    sentences = search_system.preprocess_document(document_text)

    # Generate embeddings
    embeddings = search_system.generate_embeddings(sentences)

    # Store in vector database
    search_system.store_embeddings(sentences, embeddings)


if __name__ == "__main__":
    # Example usage for initialization
    document_text = """[Your whitepaper text goes here]"""
    initialize_system(document_text)
