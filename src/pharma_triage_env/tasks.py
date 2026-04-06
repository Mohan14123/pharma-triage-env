"""
Task definitions for PharmaTriageEnv.
"""

TASKS = {
    "easy": {
        "description": "Complete, clean data with aligned signals",
        "difficulty": "low",
        "max_steps": 3,
        "expected_score_range": (0.85, 1.0),
        "features": [
            "All observation fields present",
            "Symptoms match known side effects",
            "No contradictory signals",
            "Clear free-text descriptions",
        ],
    },
    "medium": {
        "description": "Partial observability, noisy inputs, some ambiguity",
        "difficulty": "moderate",
        "max_steps": 5,
        "expected_score_range": (0.6, 0.85),
        "features": [
            "Some fields hidden (hospitalized, life_threatening may be None)",
            "Noisy free-text with abbreviations",
            "Known side effects may be incomplete",
            "Some concomitant drugs present",
            "Demographics partially available",
        ],
    },
    "hard": {
        "description": "Adversarial cases, conflicting signals, drug interactions, deceptive labels",
        "difficulty": "high",
        "max_steps": 5,
        "expected_score_range": (0.4, 0.7),
        "features": [
            "Heavy partial observability — many fields missing",
            "Contradictory free-text (e.g., 'mild' but ICU admission)",
            "Drug interactions producing emergent adverse events",
            "Deceptive expectedness flips",
            "Escalation traps",
            "OCR errors and typos in free text",
            "Ambiguous and impossible-to-judge cases",
            "Decoy side-effect lists from wrong drugs",
        ],
    },
}