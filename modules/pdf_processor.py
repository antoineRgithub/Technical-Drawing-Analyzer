"""
pdf_processor.py
Handles extraction of text and page images from PDF files using PyMuPDF.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF
from PIL import Image


@dataclass
class PDFPage:
    """Holds data extracted from a single PDF page."""

    page_number: int
    text: str
    image_b64: str  # base64-encoded PNG of the page


@dataclass
class PDFDocument:
    """Aggregated data from all pages of a PDF."""

    filename: str
    pages: List[PDFPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"--- Page {p.page_number} ---\n{p.text}" for p in self.pages
        )


def load_pdf(file_bytes: bytes, filename: str, dpi: int = 150) -> PDFDocument:
    """
    Load a PDF from raw bytes and extract text + rasterised images for every page.

    Args:
        file_bytes: Raw bytes of the PDF file.
        filename:   Original filename (used for display only).
        dpi:        Resolution used when rasterising pages to images.

    Returns:
        A PDFDocument instance populated with one PDFPage per page.
    """
    doc = PDFDocument(filename=filename)

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]

            # --- text ---
            text = page.get_text("text") or ""

            # --- image ---
            zoom = dpi / 72  # 72 dpi is the default PDF unit
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            doc.pages.append(
                PDFPage(
                    page_number=page_index + 1,
                    text=text,
                    image_b64=image_b64,
                )
            )

    return doc
