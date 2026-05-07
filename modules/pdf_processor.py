"""
pdf_processor.py
Handles extraction of text and page images from PDF files using PyMuPDF.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
@dataclass
class PDFPage:
    """Holds data extracted from a single PDF page."""

    page_number: int
    text: str
    image_b64: str  # base64-encoded PNG of the page
    img : Image.Image  # PIL image object of the page (for further processing)


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
            # text = page.get_text("text") or "no text found"

            # --- image ---
            zoom = dpi / 72  # 72 dpi is the default PDF unit
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            width, height = img.size
            drawings = img.crop((0, 0, width // 2, height))
            # right half
            right_img = img.crop((width // 2, 0, width, height))

            top_right_img = right_img.crop((0, 0, width // 2, height * 0.35))
            text1 = pytesseract.image_to_string(top_right_img)

            middle_right_img = right_img.crop((0, height * 0.35, width // 2, height * 0.85))
            text2 = pytesseract.image_to_string(middle_right_img)

            doc.pages.append(
                PDFPage(
                    page_number=page_index + 1,
                    text=text1.strip() + "\n" + text2.strip() or "no text found",
                    image_b64=image_b64,
                    img=img,
                )
            )

    return doc


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Apply image processing to enhance OCR accuracy."""
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31, 10
)
    
    # extract text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # extract drawing regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    drawing_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    text_region = cv2.bitwise_and(img_np, img_np, mask=text_mask)
    drawing_region = cv2.bitwise_and(img_np, img_np, mask=drawing_mask)

    return text_region, drawing_region