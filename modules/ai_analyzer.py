"""
ai_analyzer.py
Uses OpenAI GPT-4o (with vision) to analyse industrial technical drawings and
extract structured manufacturing information from both the text and schematic
images contained in the PDF pages.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import OpenAI

from .pdf_processor import PDFDocument

# ---------------------------------------------------------------------------
# Data model for the analysis result
# ---------------------------------------------------------------------------

@dataclass
class Dimension:
    name: str
    value: float
    unit: str
    tolerance_plus: Optional[float] = None
    tolerance_minus: Optional[float] = None

    def __str__(self) -> str:
        tol = ""
        if self.tolerance_plus is not None and self.tolerance_minus is not None:
            tol = f" +{self.tolerance_plus}/{self.tolerance_minus}"
        elif self.tolerance_plus is not None:
            tol = f" ±{self.tolerance_plus}"
        return f"{self.name}: {self.value} {self.unit}{tol}"


@dataclass
class PartAnalysis:
    part_name: str = "Unknown Part"
    material: str = "Unknown"
    dimensions: List[Dimension] = field(default_factory=list)
    surface_finish: str = "Standard"
    heat_treatment: str = "None"
    quantity: int = 1
    notes: List[str] = field(default_factory=list)
    raw_json: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert mechanical engineer specialising in CNC machining and manufacturing cost estimation.

You will receive:
1. The full text extracted from a PDF technical drawing of an industrial machined part.
2. One or more base64-encoded images of the drawing pages.

Your task is to analyse both sources and return a JSON object (and ONLY that JSON object, no markdown fences) with the following fields:

{
  "part_name": "<string>",
  "material": "<string — e.g. 'Aluminum 6061-T6', 'Steel 1045', 'Stainless Steel 316L', 'Brass C360'>",
  "quantity": <integer, default 1>,
  "surface_finish": "<string — e.g. 'Ra 1.6 µm', 'Anodised', 'Polished', 'Standard'>",
  "heat_treatment": "<string — e.g. 'None', 'Annealing', 'Quench & Temper', 'Case Hardening'>",
  "dimensions": [
    {
      "name": "<dimension name, e.g. 'Overall Length', 'Outer Diameter', 'Wall Thickness'>",
      "value": <number>,
      "unit": "<'mm' | 'cm' | 'in'>",
      "tolerance_plus": <number or null>,
      "tolerance_minus": <number or null>
    }
  ],
  "notes": ["<any important manufacturing notes, GD&T callouts, special requirements>"]
}

Rules:
- Extract ALL dimensions visible in the drawing and text.
- If a dimension appears in the text and in the image, prefer the image value.
- If a field is not determinable, use the default value shown above.
- Return valid JSON only — no extra text, no markdown code fences.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_document(
    doc: PDFDocument,
    api_key: str,
    model: str = "gpt-4o",
) -> PartAnalysis:
    """
    Send the document text and page images to an OpenAI vision model and parse
    the response into a PartAnalysis dataclass.

    Args:
        doc:     The PDFDocument to analyse.
        api_key: OpenAI API key.
        model:   OpenAI model name. Must support vision (image) inputs.
                 Tested with: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo".

    Returns:
        A PartAnalysis instance.
    """
    client = OpenAI(api_key=api_key)

    # Build message content: text first, then one image per page
    content: list = [
        {
            "type": "text",
            "text": (
                "Here is the extracted text from the PDF:\n\n"
                f"{doc.full_text}\n\n"
                "Now analyse the following drawing page images:"
            ),
        }
    ]

    for page in doc.pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{page.image_b64}",
                    "detail": "high",
                },
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=2048,
        temperature=0,
    )

    raw_text = response.choices[0].message.content or "{}"

    # Strip accidental markdown fences just in case
    raw_text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()

    try:
        data: Dict = json.loads(raw_text)
    except json.JSONDecodeError:
        # Graceful degradation: return empty analysis with the raw text as a note
        return PartAnalysis(notes=[f"Could not parse AI response: {raw_text}"])

    dimensions = [
        Dimension(
            name=d.get("name", ""),
            value=float(d.get("value", 0)),
            unit=d.get("unit", "mm"),
            tolerance_plus=_opt_float(d.get("tolerance_plus")),
            tolerance_minus=_opt_float(d.get("tolerance_minus")),
        )
        for d in data.get("dimensions", [])
    ]

    return PartAnalysis(
        part_name=data.get("part_name", "Unknown Part"),
        material=data.get("material", "Unknown"),
        dimensions=dimensions,
        surface_finish=data.get("surface_finish", "Standard"),
        heat_treatment=data.get("heat_treatment", "None"),
        quantity=int(data.get("quantity", 1)),
        notes=data.get("notes", []),
        raw_json=data,
    )


def _opt_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
