"""
Pydantic models for PharmaTriageEnv.

Typed Observation, Action, and Reward models for OpenEnv compliance.
Supports multi-step interaction with query/reveal mechanics.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Observation(BaseModel):
    """
    Structured + unstructured clinical observation for an ADE report.
    Fields may be None to simulate partial observability.
    """
    drug_name: str
    symptoms: List[str]
    hospitalized: Optional[bool] = None
    life_threatening: Optional[bool] = None
    known_label_side_effects: Optional[List[str]] = None
    free_text: Optional[str] = None

    # --- rich clinical fields ---
    patient_age: Optional[int] = None
    patient_weight_kg: Optional[float] = None
    concomitant_drugs: Optional[List[str]] = None
    medical_history: Optional[List[str]] = None
    onset_days: Optional[int] = None
    reporter_type: Optional[str] = None          # "physician", "nurse", "patient", "pharmacist"
    dechallenge_positive: Optional[bool] = None   # did stopping the drug help?
    rechallenge_positive: Optional[bool] = None   # did restarting the drug reproduce?
    lab_values: Optional[dict] = None             # e.g. {"ALT": 320, "creatinine": 2.8}

    # --- multi-step interaction ---
    step_number: int = 0
    available_queries: Optional[List[str]] = None   # queries the agent may ask
    query_response: Optional[str] = None            # env response to last query
    max_steps: int = 5


class Action(BaseModel):
    """
    Agent action: either a triage decision or an information query.
    For multi-step: set query to ask for more info before deciding.
    """
    severity: Optional[str] = None            # "low" | "medium" | "high"
    serious: Optional[bool] = None
    expected: Optional[bool] = None
    escalation: Optional[str] = None          # "routine_review" | "urgent_review" | "regulatory_report"
    query: Optional[str] = None               # info-gathering action (multi-step)


class Reward(BaseModel):
    """Shaped reward signal in range [-10, +10]."""
    value: float = Field(ge=-10.0, le=10.0)
    breakdown: Optional[dict] = None          # component-level reward breakdown