from typing import List

from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str


class SearchResult(BaseModel):
    sentence: str
    similarity_score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class PDFExtractionResponse(BaseModel):
    text: str
    num_characters: int
    status: str


class PDFProcessingAndInitResponse(BaseModel):
    extraction_status: str
    initialization_status: str
    num_characters: int
    sample_text: str
