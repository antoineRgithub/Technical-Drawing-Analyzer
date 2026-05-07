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
    # setup_cost: float = 0.0
    overhead: float = 0.0
    margin: float = 0.0
    bending_cost: float = 0.0
    holes_cost: float = 0.0
    setup_cost: float = 0.0
    total_time_hours: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.material_cost
            + self.machining_cost
            + self.setup_cost
            + self.overhead
            + self.margin
            + self.bending_cost
            + self.holes_cost
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "Matériau":            round(self.material_cost, 2),
            "Découpage":           round(self.machining_cost, 2),
            "Pliage":           round(self.bending_cost, 2),
            "Perçage":           round(self.holes_cost, 2),
            "Préparation":               round(self.setup_cost, 2),
            "Coûts indirects":            round(self.overhead, 2),
            "Marge":              round(self.margin, 2),
            "TOTAL":               round(self.total, 2),
            "Temps de découpage (h)": round(self.total_time_hours, 2),
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
    analysis: dict,
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
    density, mat_price = lookup_material(analysis['material'])
    volume_cm3 = analysis['volume'] // 1000.0  # convert from mm³ to cm³
    volume_m3 = volume_cm3 / 1_000_000.0
    mass_kg = density * volume_m3

    print(f"Estimated mass: {mass_kg:.3f} kg (density={density} kg/m³, volume={volume_cm3:.1f} cm³)")

    cb.material_cost = mass_kg * mat_price 

    # --- 2. Machining cost ---
    cutting_hours = _estimate_cutting_hours(volume_cm3)
    # finish_mult = _surface_finish_multiplier(analysis.surface_finish)
    # cb.machining_cost = cutting_hours * params.hourly_rate * finish_mult
    cb.machining_cost = cutting_hours * params.hourly_rate 

    cb.setup_cost = params.setup_time * params.hourly_rate
    

    cb.bending_cost = int(analysis['number_of_bendings']) * 5/60 * params.hourly_rate # we assume that each bending takes 5 minutes, we can adjust this value in the sidebar if needed 
    cb.holes_cost = int(analysis['number_of_holes']) * 2/60 * params.hourly_rate # we assume that each hole takes 10 minutes, we can adjust this value in the sidebar if needed 

    cb.total_time_hours = params.setup_time + cutting_hours + int(analysis['number_of_bendings']) * 5/60 + int(analysis['number_of_holes']) * 2/60 # total time is the sum of cutting time, bending time and holes time


    # --- 6. Overhead + margin ---
    subtotal = (
        cb.material_cost
        + cb.machining_cost
        + cb.bending_cost
        + cb.holes_cost
        + cb.setup_cost
    )
    cb.overhead = subtotal * params.overhead_rate
    cb.margin = subtotal * params.margin_rate
    return cb




def lookup_material(material_str: str) -> tuple[float, float]:
    """Return (density, price_per_kg) for the closest known material."""
    material_str_lower = material_str.lower()
    for key, val in MATERIAL_DB.items():
        if key.lower() in material_str_lower or material_str_lower in key.lower():
            return val
    return MATERIAL_DB["Unknown"]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _estimate_cutting_hours(volume_cm3: float) -> float:
    """
    Estimate cutting time in hours.
    Heuristic: base time proportional to cube-root of volume,
    plus a bonus for each dimension (= complexity proxy).
    """
    # Heuristic: base time scales with volume.
    base_hours = (volume_cm3 * 0.8)/60
    print(f"Estimated cutting hours: {base_hours:.2f} h (volume={volume_cm3:.1f} cm³)")
    return base_hours


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
