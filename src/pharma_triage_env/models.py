from pydantic import BaseModel
from typing import List, Optional

class Observation(BaseModel):
    drug_name: str
    symptoms: List[str]
    hospitalized: Optional[bool]
    life_threatening: Optional[bool]
    known_label_side_effects: Optional[List[str]] 

class Action(BaseModel):
    severity: str
    serious: bool
    expected: bool
    escalation: str

class Reward(BaseModel):
    value: float