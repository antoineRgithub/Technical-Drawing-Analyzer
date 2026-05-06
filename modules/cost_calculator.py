"""
cost_calculator.py
Estimates the manufacturing cost of a machined part from its PartAnalysis.

The model is intentionally transparent and configurable so the user can
adjust rates inside the Streamlit sidebar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .ai_analyzer import PartAnalysis

# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------

# Each entry: material_key -> (density_kg_per_m3, price_euros_per_kg)
MATERIAL_DB: Dict[str, tuple[float, float]] = {
    "Aluminum":         (2700.0,  4.0),
    "Aluminum 6061":    (2700.0,  4.5),
    "Aluminum 7075":    (2810.0,  7.0),
    "Steel":            (7850.0,  1.0),
    "Steel 1045":       (7850.0,  1.2),
    "Stainless Steel":  (8000.0,  5.0),
    "Stainless Steel 316L": (8000.0, 6.0),
    "Stainless Steel 304":  (7930.0, 5.5),
    "Brass":            (8500.0,  7.0),
    "Brass C360":       (8500.0,  7.5),
    "Titanium":         (4500.0, 35.0),
    "Unknown":          (7850.0,  2.0),  # fallback to steel
}

# Surface finish surcharge multiplier on machining cost
SURFACE_FINISH_MULTIPLIER: Dict[str, float] = {
    "Standard": 1.0,
    "Ra 3.2":   1.0,
    "Ra 1.6":   1.15,
    "Ra 0.8":   1.35,
    "Ra 0.4":   1.60,
    "Polished": 1.80,
    "Anodised": 1.20,
    "Hard Anodised": 1.40,
}

# Heat-treatment surcharge (flat €)
HEAT_TREATMENT_COST: Dict[str, float] = {
    "None":             0.0,
    "Annealing":        25.0,
    "Quench & Temper":  45.0,
    "Case Hardening":   60.0,
    "Nitriding":        80.0,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    material_cost: float = 0.0
    machining_cost: float = 0.0
    setup_cost: float = 0.0
    surface_treatment_cost: float = 0.0
    heat_treatment_cost: float = 0.0
    overhead_and_margin: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.material_cost
            + self.machining_cost
            + self.setup_cost
            + self.surface_treatment_cost
            + self.heat_treatment_cost
            + self.overhead_and_margin
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "Material":            round(self.material_cost, 2),
            "Machining":           round(self.machining_cost, 2),
            "Setup":               round(self.setup_cost, 2),
            "Surface Treatment":   round(self.surface_treatment_cost, 2),
            "Heat Treatment":      round(self.heat_treatment_cost, 2),
            "Overhead & Margin":   round(self.overhead_and_margin, 2),
            "TOTAL":               round(self.total, 2),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CostParameters:
    """User-adjustable parameters injected from the Streamlit sidebar."""
    hourly_rate: float = 65.0         # €/h — CNC machining rate
    setup_time: float = 0.5           # h — fixed setup time
    overhead_rate: float = 0.25       # fraction of (material + machining) added as overhead
    margin_rate: float = 0.15         # profit margin fraction on top of all costs


def estimate_cost(
    analysis: PartAnalysis,
    params: Optional[CostParameters] = None,
) -> CostBreakdown:
    """
    Compute a transparent cost breakdown from the extracted part data.

    The machining time is estimated from the bounding-box volume and number
    of dimensions (as a proxy for geometric complexity).
    """
    if params is None:
        params = CostParameters()

    cb = CostBreakdown()

    # --- 1. Material cost ---
    density, mat_price = _lookup_material(analysis.material)
    volume_cm3 = _estimate_volume_cm3(analysis)
    volume_m3 = volume_cm3 / 1_000_000.0
    mass_kg = density * volume_m3

    # Add 20 % stock allowance
    cb.material_cost = mass_kg * mat_price * 1.20

    # --- 2. Machining cost ---
    machining_hours = _estimate_machining_hours(analysis, volume_cm3)
    finish_mult = _surface_finish_multiplier(analysis.surface_finish)
    cb.machining_cost = machining_hours * params.hourly_rate * finish_mult

    # --- 3. Setup cost ---
    cb.setup_cost = params.setup_time * params.hourly_rate

    # --- 4. Surface treatment ---
    cb.surface_treatment_cost = _surface_treatment_cost(analysis.surface_finish)

    # --- 5. Heat treatment ---
    cb.heat_treatment_cost = _heat_treatment_cost(analysis.heat_treatment)

    # --- 6. Overhead + margin ---
    subtotal = (
        cb.material_cost
        + cb.machining_cost
        + cb.setup_cost
        + cb.surface_treatment_cost
        + cb.heat_treatment_cost
    )
    cb.overhead_and_margin = subtotal * (params.overhead_rate + params.margin_rate)

    return cb


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _lookup_material(material_str: str) -> tuple[float, float]:
    """Return (density, price_per_kg) for the closest known material."""
    material_str_lower = material_str.lower()
    for key, val in MATERIAL_DB.items():
        if key.lower() in material_str_lower or material_str_lower in key.lower():
            return val
    return MATERIAL_DB["Unknown"]


def _estimate_volume_cm3(analysis: PartAnalysis) -> float:
    """
    Estimate the bounding-box volume (cm³) of the part from extracted dimensions.
    Converts all units to mm before computing volume.
    """
    dims_mm = {}
    for dim in analysis.dimensions:
        val_mm = _to_mm(dim.value, dim.unit)
        name_lower = dim.name.lower()
        # Try to identify length/width/height/diameter dimensions
        if any(k in name_lower for k in ("length", "long", "hauteur", "height", "overall")):
            dims_mm.setdefault("length", val_mm)
        elif any(k in name_lower for k in ("width", "larg", "largeur")):
            dims_mm.setdefault("width", val_mm)
        elif any(k in name_lower for k in ("depth", "profond", "thickness", "épaisseur", "wall")):
            dims_mm.setdefault("depth", val_mm)
        elif any(k in name_lower for k in ("diameter", "diam", "ø", "od", "outer")):
            dims_mm.setdefault("diameter", val_mm)
        elif any(k in name_lower for k in ("radius", "rayon")):
            dims_mm.setdefault("diameter", val_mm * 2)

    if not dims_mm:
        # No dimension data — return a small default volume (50×50×50 mm)
        return 125.0

    if "diameter" in dims_mm:
        r = dims_mm["diameter"] / 2.0
        h = dims_mm.get("length", dims_mm.get("depth", dims_mm["diameter"]))
        volume_mm3 = math.pi * r * r * h
    else:
        length = dims_mm.get("length", 50.0)
        width = dims_mm.get("width", length)
        depth = dims_mm.get("depth", min(length, width) / 2.0)
        volume_mm3 = length * width * depth

    return volume_mm3 / 1000.0  # mm³ → cm³


def _estimate_machining_hours(analysis: PartAnalysis, volume_cm3: float) -> float:
    """
    Estimate machining time in hours.
    Heuristic: base time proportional to cube-root of volume,
    plus a bonus for each dimension (= complexity proxy).
    """
    # Heuristic: base time scales with cube-root of volume (surface-area proxy),
    # plus 0.05 h per dimension to account for geometric complexity (features count).
    # The leading coefficient 0.1 is calibrated so a 125 cm³ block ≈ 0.5 h base time.
    base_hours = 0.1 * (volume_cm3 ** (1 / 3))
    complexity_bonus = len(analysis.dimensions) * 0.05  # 3 min per additional feature
    return max(0.2, base_hours + complexity_bonus)


def _to_mm(value: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in ("cm",):
        return value * 10.0
    if unit in ("in", "inch", "inches"):
        return value * 25.4
    return value  # already mm


def _surface_finish_multiplier(surface_finish: str) -> float:
    finish_lower = surface_finish.lower()
    for key, mult in SURFACE_FINISH_MULTIPLIER.items():
        if key.lower() in finish_lower:
            return mult
    return 1.0


def _surface_treatment_cost(surface_finish: str) -> float:
    finish_lower = surface_finish.lower()
    if "anodis" in finish_lower:
        return 15.0
    if "polish" in finish_lower:
        return 20.0
    if "paint" in finish_lower or "peinture" in finish_lower:
        return 10.0
    if "zinc" in finish_lower or "galvan" in finish_lower:
        return 12.0
    return 0.0


def _heat_treatment_cost(heat_treatment: str) -> float:
    ht_lower = heat_treatment.lower()
    for key, cost in HEAT_TREATMENT_COST.items():
        if key.lower() in ht_lower:
            return cost
    return 0.0
