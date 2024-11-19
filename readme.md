# Document Similarity Search System

This system allows users to search for the most similar text sentences from documents based on a query input. It uses vector embeddings and Pinecone vector database for efficient similarity search.

## Features

- PDF document processing
- Text similarity search using vector embeddings
- RESTful API with FastAPI
- Docker support for easy deployment

## Prerequisites

- Python 3.11+
- Docker (optional)
- Pinecone API key (sign up at [Pinecone](https://www.pinecone.io/))

## Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create `.env` file and add your Pinecone credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

3. Build and run with Docker Compose:
```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

### Local Development

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file and add your Pinecone credentials:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Usage

The API provides two main endpoints:

### 1. Add Document

Upload a PDF document to be processed and indexed:

```bash
curl -X POST "http://localhost:8000/add_pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf"
```

Python example:
```python
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:8000/add_pdf', files=files)
print(response.json())
```

### 2. Search Similar Sentences

Search for sentences similar to your query:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is tactical maturity?"}'
```

Python example:
```python
import requests

query = {"query": "What is tactical maturity?"}
response = requests.post('http://localhost:8000/search', json=query)
print(response.json())
```

## API Documentation

Once the server is running, you can access:
- API documentation: `http://localhost:8000/docs`
- Alternative documentation: `http://localhost:8000/redoc`

## Project Structure

```
similarity-search/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── pdf_processor.py
│       └── similarity_search.py
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## Environment Variables

Required environment variables:
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `PINECONE_INDEX_NAME`: Name for your Pinecone index (default: "document-search")
