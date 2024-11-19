import io
from typing import List
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """
        Extract text from a PDF file content.

        Args:
            file_content (bytes): The binary content of the PDF file

        Returns:
            str: Extracted text from the PDF
        """
        try:
            pdf_reader = PdfReader(io.BytesIO(file_content))

            text_content = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)

            full_text = "\n".join(text_content)

            cleaned_text = "\n".join(line for line in full_text.splitlines() if line.strip())

            return cleaned_text

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")

    @staticmethod
    def extract_text_from_pdfs(files: List[bytes]) -> str:
        """
        Extract text from multiple PDF files and combine them.

        Args:
            files (List[bytes]): List of PDF file contents

        Returns:
            str: Combined extracted text from all PDFs
        """
        all_texts = []
        for file_content in files:
            text = PDFProcessor.extract_text_from_pdf(file_content)
            all_texts.append(text)

        return "\n\n".join(all_texts)
