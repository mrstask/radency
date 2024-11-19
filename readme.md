# Document Similarity Search System

This system implements a similarity search functionality that allows users to find the most similar sentences from a whitepaper based on a query input. It uses sentence transformers for embedding generation and Pinecone as the vector database.

## Prerequisites

- Python 3.8+
- Pinecone API key
- Docker (optional)

## Installation

1. Clone the repository and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install sentence-transformers pinecone-client fastapi uvicorn python-dotenv nltk
```

3. Create a `.env` file in the project root with your Pinecone credentials:
```
PINECONE_API_KEY=your_api_key_here
PINECONE_ENV=your_environment_here
```

## Running the Application

1. Initialize the system with your document:
```python
from main import initialize_system

with open('whitepaper.txt', 'r') as f:
    document_text = f.read()
initialize_system(document_text)
```

2. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

3. The API will be available at `http://localhost:8000`

## Using Docker

1. Build the Docker image:
```bash
docker build -t similarity-search .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e PINECONE_API_KEY=your_key -e PINECONE_ENV=your_env similarity-search
```

## API Usage

Send a POST request to `/search` endpoint with your query:

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the characteristics of tactical maturity?"}'
```

The response will contain the top 3 most similar sentences and their similarity scores:

```json
[
    {
        "sentence": "Organizations at the tactical phase are exploring the potential of AI to deliver in the short term.",
        "similarity_score": 0.85
    },
    {
        "sentence": "Use cases tend to be narrow, and developers are typically leveraging exploratory data analysis (EDA) tools and ready-to-use AI and ML services for proofs of concept and prototyping.",
        "similarity_score": 0.78
    },
    {
        "sentence": "At this phase, organizations are aware of the promise of advanced analytics, but ML can be seen as unattainable, and for this reason, complex problems are outsourced.",
        "similarity_score": 0.72
    }
]
```

## Project Structure

- `main.py`: Main application file containing the SimilaritySearchSystem class and FastAPI endpoints
- `Dockerfile`: Docker configuration for containerization
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not included in repository)
