import logging
import os

import nltk
from fastapi import FastAPI, HTTPException, UploadFile, File

from app.models.schemas import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    PDFExtractionResponse,
    PDFProcessingAndInitResponse
)
from app.services.pdf_processor import PDFProcessor
from app.services.similarity_search import SimilaritySearchSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_nltk():
    """Initialize NLTK by downloading required resources."""
    try:
        # Set NLTK data path to a directory in the project
        nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        # Download required NLTK data
        required_packages = ['punkt', 'punkt_tab']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, download_dir=nltk_data_dir)

        logger.info("NLTK initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing NLTK: {str(e)}")
        raise


initialize_nltk()

app = FastAPI(title="Document Similarity Search API")

search_system = SimilaritySearchSystem()
pdf_processor = PDFProcessor()


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(query: SearchQuery) -> SearchResponse:
    """Search endpoint that returns top similar sentences."""
    try:
        results = search_system.search(query.query)
        return SearchResponse(
            results=[
                SearchResult(sentence=sent, similarity_score=score)
                for sent, score in results
            ]
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_pdf", response_model=PDFProcessingAndInitResponse)
async def process_and_init_pdf_endpoint(file: UploadFile = File(...)) -> PDFProcessingAndInitResponse:
    """Extract text from a PDF file and initialize the search system with it."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        content = await file.read()
        extracted_text = pdf_processor.extract_text_from_pdf(content)
        search_system.initialize_with_document(extracted_text)

        return PDFProcessingAndInitResponse(
            extraction_status="success",
            initialization_status="success",
            num_characters=len(extracted_text),
            sample_text=extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        )
    except Exception as e:
        logger.error(f"PDF processing and initialization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
