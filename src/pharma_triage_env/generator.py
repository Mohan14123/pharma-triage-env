"""
Case generator for PharmaTriageEnv.

Produces complex, realistic ADE cases with:
- Real drug/symptom combinations
- Drug interactions and comorbidities
- Real-world data artifacts (typos, abbreviations, OCR errors)
- Ambiguous / impossible / partial / noisy / time-sensitive cases
- Multi-step hidden information that can be revealed via queries
- Deterministic ground truth for grading

Difficulty tiers:
  EASY   – complete clean data, aligned signals
  MEDIUM – partial observability, noisy inputs, some ambiguity
  HARD   – adversarial, conflicting signals, deceptive labels,
            drug interactions, multi-agent artifacts
"""

import random
import math

# Module-level default RNG (can be overridden per-call via seed)
_default_rng = random.Random()

# ============================================================
# DRUG DATABASE (realistic, FAERS-inspired)
# ============================================================

DRUGS = {
    "Metformin":     {"class": "antidiabetic",    "common_se": ["nausea", "diarrhea", "lactic acidosis"],                "rare_se": ["vitamin B12 deficiency"]},
    "Lisinopril":    {"class": "ACE inhibitor",   "common_se": ["dry cough", "dizziness", "hyperkalemia"],              "rare_se": ["angioedema"]},
    "Atorvastatin":  {"class": "statin",          "common_se": ["muscle pain", "headache", "nausea"],                   "rare_se": ["rhabdomyolysis", "hepatotoxicity"]},
    "Warfarin":      {"class": "anticoagulant",   "common_se": ["bruising", "bleeding gums", "nosebleed"],              "rare_se": ["hemorrhagic stroke", "skin necrosis"]},
    "Amoxicillin":   {"class": "antibiotic",      "common_se": ["rash", "diarrhea", "nausea"],                          "rare_se": ["anaphylaxis", "Stevens-Johnson syndrome"]},
    "Omeprazole":    {"class": "PPI",             "common_se": ["headache", "nausea", "abdominal pain"],                "rare_se": ["C. difficile infection", "hypomagnesemia"]},
    "Sertraline":    {"class": "SSRI",            "common_se": ["insomnia", "nausea", "dizziness", "dry mouth"],        "rare_se": ["serotonin syndrome", "suicidal ideation"]},
    "Amlodipine":    {"class": "calcium blocker", "common_se": ["peripheral edema", "dizziness", "flushing"],           "rare_se": ["severe hypotension"]},
    "Ciprofloxacin": {"class": "fluoroquinolone", "common_se": ["nausea", "diarrhea", "headache"],                      "rare_se": ["tendon rupture", "QT prolongation", "aortic dissection"]},
    "Carbamazepine": {"class": "anticonvulsant",  "common_se": ["drowsiness", "dizziness", "nausea"],                   "rare_se": ["aplastic anemia", "Stevens-Johnson syndrome", "DRESS syndrome"]},
    "Methotrexate":  {"class": "immunosuppressant","common_se": ["nausea", "fatigue", "mouth sores"],                   "rare_se": ["pancytopenia", "pulmonary fibrosis", "hepatotoxicity"]},
    "Ibuprofen":     {"class": "NSAID",           "common_se": ["stomach pain", "nausea", "headache"],                  "rare_se": ["GI bleeding", "renal failure", "myocardial infarction"]},
}

# Drug interaction pairs that produce emergent adverse events
DRUG_INTERACTIONS = [
    {"drugs": ["Warfarin", "Ibuprofen"],       "emergent": "GI bleeding",         "severity": "high"},
    {"drugs": ["Sertraline", "Methotrexate"],  "emergent": "serotonin syndrome",  "severity": "high"},
    {"drugs": ["Lisinopril", "Ibuprofen"],     "emergent": "renal failure",       "severity": "high"},
    {"drugs": ["Ciprofloxacin", "Warfarin"],   "emergent": "hemorrhagic event",   "severity": "high"},
    {"drugs": ["Amlodipine", "Atorvastatin"],  "emergent": "rhabdomyolysis",      "severity": "medium"},
    {"drugs": ["Carbamazepine", "Warfarin"],   "emergent": "subtherapeutic INR",  "severity": "medium"},
    {"drugs": ["Omeprazole", "Methotrexate"],  "emergent": "methotrexate toxicity","severity": "high"},
]

COMORBIDITIES = [
    "diabetes", "hypertension", "chronic kidney disease", "liver cirrhosis",
    "heart failure", "asthma", "epilepsy", "depression", "rheumatoid arthritis",
    "atrial fibrillation", "COPD", "obesity"
]

REPORTER_TYPES = ["physician", "nurse", "patient", "pharmacist"]

# ============================================================
# NOISE / ARTIFACT HELPERS
# ============================================================

TYPO_MAP = {
    "rash": ["rsh", "rahs", "rah"],
    "nausea": ["nasea", "nausae", "nusea"],
    "diarrhea": ["diarrhoea", "diahrea", "diarreha"],
    "headache": ["headach", "hedache", "headake"],
    "anaphylaxis": ["anaphylaxsis", "anaphillaxis", "anaphlaxis"],
    "dizziness": ["dizzines", "dizzness", "dizzyness"],
    "bleeding": ["bleading", "bleding"],
    "hospitalized": ["hospitalised", "hosptalized", "hospitlized"],
    "Stevens-Johnson syndrome": ["SJS", "Steven Johnson", "steven-johnson syn"],
}

ABBREVIATIONS = {
    "hospitalized": "hosp", "patient": "pt", "medication": "med",
    "symptoms": "sx", "diagnosis": "dx", "treatment": "tx",
    "history": "hx", "laboratory": "lab", "prescription": "Rx",
    "discontinuation": "d/c", "intravenous": "IV",
}

OCR_ERRORS = {
    "l": "1", "I": "l", "O": "0", "o": "0",
    "S": "5", "B": "8", "G": "6", "Z": "2",
}

NOISE_PHRASES = [
    "Patient unsure about onset timeline",
    "Symptoms may be unrelated to medication",
    "Reporter notes are partially illegible",
    "Report received via fax — some fields unclear",
    "Translated from original report — possible interpretation errors",
    "Follow-up information pending",
    "Concomitant medication list incomplete",
    "Patient has poor recall of events",
    "Multiple reporters provided conflicting timelines",
    "Initial assessment was revised after lab results",
    "Some data redacted for privacy",
    "Duplicate report suspected — under review",
]

MISLEADING_PHRASES = [
    "Symptoms appear mild despite clinical complications.",
    "Patient reports feeling fine but labs are abnormal.",
    "Event unlikely to be drug-related per initial assessment.",
    "Treating physician does not suspect the medication.",
    "Similar symptoms reported before starting the drug.",
    "Event occurred long after drug discontinuation.",
    "Patient has a history of similar unrelated events.",
]

CONTRADICTORY_PHRASES = [
    "Report states patient was NOT hospitalized, but discharge summary attached.",
    "Physician notes mild reaction; ER records indicate ICU admission.",
    "Patient denies life-threatening event; vitals show critical status.",
    "Drug label lists symptom as common but reporter calls it unexpected.",
]


def _apply_typos(text, prob=0.15, rng=None):
    """Inject realistic typos into text."""
    rng = rng or _default_rng
    words = text.split()
    result = []
    for w in words:
        lower = w.lower()
        if lower in TYPO_MAP and rng.random() < prob:
            result.append(rng.choice(TYPO_MAP[lower]))
        else:
            result.append(w)
    return " ".join(result)


def _apply_ocr_errors(text, prob=0.08, rng=None):
    """Simulate OCR transcription errors."""
    rng = rng or _default_rng
    chars = list(text)
    for i, c in enumerate(chars):
        if c in OCR_ERRORS and rng.random() < prob:
            chars[i] = OCR_ERRORS[c]
    return "".join(chars)


def _apply_abbreviations(text, prob=0.2, rng=None):
    """Replace words with medical abbreviations."""
    rng = rng or _default_rng
    for full, abbrev in ABBREVIATIONS.items():
        if full in text.lower() and rng.random() < prob:
            text = text.replace(full, abbrev)
    return text


def maybe_missing(value, prob=0.25, rng=None):
    """Return None with given probability (partial observability)."""
    rng = rng or _default_rng
    return None if rng.random() < prob else value


# ============================================================
# LAB VALUE GENERATOR
# ============================================================

def _generate_labs(serious, drug_name, rng=None):
    """Generate plausible lab values with optional abnormalities."""
    rng = rng or _default_rng
    labs = {}
    drug_info = DRUGS.get(drug_name, {})

    # baseline labs
    alt_base = rng.randint(10, 40)
    creatinine_base = round(rng.uniform(0.6, 1.2), 1)
    hemoglobin_base = round(rng.uniform(12.0, 16.0), 1)
    platelet_base = rng.randint(150, 400)
    inr_base = round(rng.uniform(0.9, 1.1), 1)

    if serious:
        # abnormal labs for serious cases
        if "hepatotoxicity" in drug_info.get("rare_se", []):
            alt_base = rng.randint(200, 800)
        if "renal failure" in drug_info.get("rare_se", []):
            creatinine_base = round(rng.uniform(3.0, 8.0), 1)
        if "pancytopenia" in drug_info.get("rare_se", []):
            platelet_base = rng.randint(10, 50)
            hemoglobin_base = round(rng.uniform(5.0, 8.0), 1)
        if drug_info.get("class") == "anticoagulant":
            inr_base = round(rng.uniform(4.0, 9.0), 1)

    labs["ALT"] = alt_base
    labs["creatinine"] = creatinine_base
    labs["hemoglobin"] = hemoglobin_base
    labs["platelets"] = platelet_base
    if drug_info.get("class") == "anticoagulant" or rng.random() < 0.3:
        labs["INR"] = inr_base
    if rng.random() < 0.4:
        labs["WBC"] = round(rng.uniform(2.0, 15.0), 1)
    if rng.random() < 0.3:
        labs["potassium"] = round(rng.uniform(3.0, 6.5), 1)

    return labs


# ============================================================
# HIDDEN INFO FOR MULTI-STEP QUERIES
# ============================================================

QUERY_CATALOG = [
    "was_patient_hospitalized",
    "was_event_life_threatening",
    "what_are_concomitant_drugs",
    "what_is_medical_history",
    "was_dechallenge_positive",
    "was_rechallenge_positive",
    "what_are_lab_values",
    "what_is_reporter_type",
    "what_is_onset_timeline",
    "is_symptom_in_drug_label",
]


def _build_hidden_info(case_data, gt):
    """Build the hidden-info dict that the env reveals on query."""
    hidden = {}

    obs = case_data["observation"]
    hidden["was_patient_hospitalized"] = {
        "field": "hospitalized",
        "value": gt.get("_true_hospitalized", obs.get("hospitalized")),
        "response": f"Records confirm patient {'was' if gt.get('_true_hospitalized', obs.get('hospitalized')) else 'was NOT'} hospitalized."
    }
    hidden["was_event_life_threatening"] = {
        "field": "life_threatening",
        "value": gt.get("_true_life_threatening", obs.get("life_threatening")),
        "response": f"Clinical records indicate event {'was' if gt.get('_true_life_threatening', obs.get('life_threatening')) else 'was NOT'} life-threatening."
    }
    hidden["what_are_concomitant_drugs"] = {
        "field": "concomitant_drugs",
        "value": obs.get("concomitant_drugs"),
        "response": f"Patient was also taking: {', '.join(obs.get('concomitant_drugs') or ['none documented'])}."
    }
    hidden["what_is_medical_history"] = {
        "field": "medical_history",
        "value": obs.get("medical_history"),
        "response": f"Medical history: {', '.join(obs.get('medical_history') or ['none documented'])}."
    }
    hidden["was_dechallenge_positive"] = {
        "field": "dechallenge_positive",
        "value": obs.get("dechallenge_positive"),
        "response": f"Dechallenge was {'positive (symptoms resolved)' if obs.get('dechallenge_positive') else 'negative or not performed'}."
    }
    hidden["was_rechallenge_positive"] = {
        "field": "rechallenge_positive",
        "value": obs.get("rechallenge_positive"),
        "response": f"Rechallenge was {'positive (symptoms recurred)' if obs.get('rechallenge_positive') else 'negative or not performed'}."
    }
    hidden["what_are_lab_values"] = {
        "field": "lab_values",
        "value": obs.get("lab_values"),
        "response": f"Lab results: {obs.get('lab_values') or 'not available'}."
    }
    hidden["what_is_reporter_type"] = {
        "field": "reporter_type",
        "value": obs.get("reporter_type"),
        "response": f"Report submitted by: {obs.get('reporter_type') or 'unknown'}."
    }
    hidden["what_is_onset_timeline"] = {
        "field": "onset_days",
        "value": obs.get("onset_days"),
        "response": f"Onset was {obs.get('onset_days', 'unknown')} days after starting the drug."
    }
    kse = obs.get("known_label_side_effects") or []
    syms = obs.get("symptoms", [])
    overlap = bool(set(syms) & set(kse))
    hidden["is_symptom_in_drug_label"] = {
        "field": "known_label_side_effects",
        "value": kse,
        "response": f"Drug label side effects: {', '.join(kse) if kse else 'none listed'}. {'Overlap with reported symptoms found.' if overlap else 'No overlap with reported symptoms.'}"
    }

    return hidden


# ============================================================
# CASE GENERATOR
# ============================================================

SEVERE_SYMPTOMS = [
    "anaphylaxis", "cardiac arrest", "hemorrhagic stroke",
    "Stevens-Johnson syndrome", "serotonin syndrome", "rhabdomyolysis",
    "pulmonary fibrosis", "aplastic anemia", "DRESS syndrome",
    "aortic dissection", "pancytopenia", "GI bleeding",
    "renal failure", "hepatotoxicity", "skin necrosis",
    "tendon rupture", "QT prolongation",
]


def generate_case(task="hard", seed=None):
    """
    Generate a single ADE triage case.

    Args:
        task: "easy", "medium", or "hard"
        seed: optional RNG seed for reproducibility

    Returns:
        dict with keys: observation, ground_truth, hidden_info, case_metadata
    """
    # Create an isolated RNG instance for full reproducibility.
    # This avoids polluting the global random state.
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = _default_rng

    # ---- pick primary drug ----
    drug_name = rng.choice(list(DRUGS.keys()))
    drug_info = DRUGS[drug_name]

    # ---- concomitant drugs (interactions) ----
    concomitant_drugs = []
    interaction = None
    if task in ("medium", "hard"):
        other_drugs = [d for d in DRUGS if d != drug_name]
        num_concomitant = rng.randint(0, 3)
        concomitant_drugs = rng.sample(other_drugs, min(num_concomitant, len(other_drugs)))

        # check for known interactions
        all_patient_drugs = [drug_name] + concomitant_drugs
        for ix in DRUG_INTERACTIONS:
            if all(d in all_patient_drugs for d in ix["drugs"]):
                interaction = ix
                break

    # ---- symptoms ----
    num_symptoms = rng.randint(1, 3)

    if task == "easy":
        # easy: usually common side effects, but ~30% chance of including
        # a rare side effect to produce diverse GT patterns
        if rng.random() < 0.3:
            pool = drug_info["common_se"] + drug_info["rare_se"]
        else:
            pool = drug_info["common_se"]
        symptoms = rng.sample(pool, min(num_symptoms, len(pool)))
    elif task == "medium":
        pool = drug_info["common_se"] + drug_info["rare_se"]
        symptoms = rng.sample(pool, min(num_symptoms, len(pool)))
    else:
        # hard: may include rare + interaction-emergent symptoms
        pool = drug_info["common_se"] + drug_info["rare_se"]
        if interaction:
            pool.append(interaction["emergent"])
        symptoms = rng.sample(pool, min(num_symptoms, len(pool)))
        # 40% chance of injecting a severe symptom
        if rng.random() < 0.4 and not any(s in SEVERE_SYMPTOMS for s in symptoms):
            symptoms.append(rng.choice(SEVERE_SYMPTOMS))

    symptoms = list(set(symptoms))  # deduplicate

    # ---- clinical flags (true values) ----
    has_severe = any(s in SEVERE_SYMPTOMS for s in symptoms)
    true_hospitalized = has_severe or rng.random() < 0.35
    true_life_threatening = has_severe and rng.random() < 0.5

    if task == "easy":
        # diverse: ~55% hospitalized, life-threatening if severe or 15% chance
        true_hospitalized = has_severe or rng.random() < 0.55
        true_life_threatening = has_severe or (true_hospitalized and rng.random() < 0.15)

    if task == "hard" and rng.random() < 0.3:
        # deceptive: life-threatening but reported as not-hospitalized
        true_hospitalized = False
        true_life_threatening = True

    # ---- observable flags (may be hidden/None) ----
    hospitalized = true_hospitalized
    life_threatening = true_life_threatening

    if task == "medium":
        hospitalized = maybe_missing(hospitalized, 0.25, rng=rng)
        life_threatening = maybe_missing(life_threatening, 0.25, rng=rng)
    elif task == "hard":
        hospitalized = maybe_missing(hospitalized, 0.4, rng=rng)
        life_threatening = maybe_missing(life_threatening, 0.4, rng=rng)

    # ---- known label side effects ----
    if task == "easy":
        # diversify: 70% full list, 30% common-only (makes expected=False
        # possible when symptoms contain a rare side effect)
        if rng.random() < 0.3:
            known_side_effects = drug_info["common_se"][:]
        else:
            known_side_effects = list(set(drug_info["common_se"] + drug_info["rare_se"]))
    elif task == "medium":
        known_side_effects = drug_info["common_se"][:]
        if rng.random() < 0.3:
            known_side_effects = []
    else:
        # hard: may give misleading / empty / shuffled list
        if rng.random() < 0.25:
            known_side_effects = None
        elif rng.random() < 0.3:
            # decoy: side effects from a DIFFERENT drug
            decoy_drug = rng.choice([d for d in DRUGS if d != drug_name])
            known_side_effects = DRUGS[decoy_drug]["common_se"][:]
        else:
            known_side_effects = drug_info["common_se"][:rng.randint(1, len(drug_info["common_se"]))]

    # ---- patient demographics ----
    patient_age = rng.randint(18, 90)
    patient_weight = round(rng.uniform(45.0, 140.0), 1)
    medical_history = rng.sample(COMORBIDITIES, k=rng.randint(0, 3)) if rng.random() < 0.7 else []
    onset_days = rng.randint(1, 180)
    reporter_type = rng.choice(REPORTER_TYPES)
    dechallenge_positive = rng.choice([True, False, None])
    rechallenge_positive = rng.choice([True, False, None])

    # ---- lab values ----
    serious_true = true_hospitalized or true_life_threatening
    labs = _generate_labs(serious_true, drug_name, rng=rng)

    # ---- partial observability for demographics ----
    if task == "easy":
        obs_age = patient_age
        obs_weight = patient_weight
        obs_concomitant = concomitant_drugs if concomitant_drugs else None
        obs_history = medical_history if medical_history else None
        obs_onset = onset_days
        obs_reporter = reporter_type
        obs_dechallenge = dechallenge_positive
        obs_rechallenge = rechallenge_positive
        obs_labs = labs
    elif task == "medium":
        obs_age = maybe_missing(patient_age, 0.2, rng=rng)
        obs_weight = maybe_missing(patient_weight, 0.3, rng=rng)
        obs_concomitant = maybe_missing(concomitant_drugs if concomitant_drugs else None, 0.3, rng=rng)
        obs_history = maybe_missing(medical_history if medical_history else None, 0.3, rng=rng)
        obs_onset = maybe_missing(onset_days, 0.2, rng=rng)
        obs_reporter = maybe_missing(reporter_type, 0.2, rng=rng)
        obs_dechallenge = maybe_missing(dechallenge_positive, 0.4, rng=rng)
        obs_rechallenge = maybe_missing(rechallenge_positive, 0.5, rng=rng)
        obs_labs = maybe_missing(labs, 0.3, rng=rng)
    else:
        obs_age = maybe_missing(patient_age, 0.4, rng=rng)
        obs_weight = maybe_missing(patient_weight, 0.5, rng=rng)
        obs_concomitant = maybe_missing(concomitant_drugs if concomitant_drugs else None, 0.5, rng=rng)
        obs_history = maybe_missing(medical_history if medical_history else None, 0.5, rng=rng)
        obs_onset = maybe_missing(onset_days, 0.4, rng=rng)
        obs_reporter = maybe_missing(reporter_type, 0.3, rng=rng)
        obs_dechallenge = maybe_missing(dechallenge_positive, 0.6, rng=rng)
        obs_rechallenge = maybe_missing(rechallenge_positive, 0.7, rng=rng)
        obs_labs = maybe_missing(labs, 0.5, rng=rng)

    # ---- free text (with artifacts) ----
    base_text = f"Patient ({patient_age}y, {patient_weight}kg) reports {', '.join(symptoms)} after taking {drug_name}."
    if concomitant_drugs:
        base_text += f" Also taking {', '.join(concomitant_drugs)}."
    if medical_history:
        base_text += f" Past history: {', '.join(medical_history)}."

    if task == "easy":
        free_text = base_text
        if true_hospitalized:
            free_text += " Patient was admitted for observation."
        if true_life_threatening:
            free_text += " Event classified as life-threatening."
    elif task == "medium":
        free_text = base_text + " " + rng.choice(NOISE_PHRASES)
        if rng.random() < 0.3:
            free_text = _apply_abbreviations(free_text, rng=rng)
    else:
        # hard: maximum noise
        free_text = base_text
        free_text += " " + rng.choice(NOISE_PHRASES)
        if rng.random() < 0.5:
            free_text += " " + rng.choice(MISLEADING_PHRASES)
        if rng.random() < 0.4:
            free_text += " " + rng.choice(CONTRADICTORY_PHRASES)
        # apply artifacts
        if rng.random() < 0.3:
            free_text = _apply_typos(free_text, prob=0.12, rng=rng)
        if rng.random() < 0.2:
            free_text = _apply_ocr_errors(free_text, prob=0.06, rng=rng)
        if rng.random() < 0.3:
            free_text = _apply_abbreviations(free_text, prob=0.25, rng=rng)

    # ---- ground truth ----
    serious = true_hospitalized or true_life_threatening

    if known_side_effects is not None:
        expected = any(s in known_side_effects for s in symptoms)
    else:
        expected = any(s in (drug_info["common_se"] + drug_info["rare_se"]) for s in symptoms)

    # hard: deceptive expectedness flip
    if task == "hard" and rng.random() < 0.35:
        expected = not expected

    # severity from clinical evidence
    if any(s in SEVERE_SYMPTOMS for s in symptoms):
        severity = "high"
    elif serious:
        severity = "medium"
    else:
        severity = "low"

    # escalation
    if serious and not expected:
        escalation = "regulatory_report"
    elif serious:
        escalation = "urgent_review"
    else:
        escalation = "routine_review"

    # hard: escalation traps
    if task == "hard" and rng.random() < 0.25:
        # flip escalation to create trap
        choices = ["routine_review", "urgent_review", "regulatory_report"]
        choices.remove(escalation)
        escalation = rng.choice(choices)

    # ---- determine available queries ----
    available = list(QUERY_CATALOG)
    if task == "easy":
        available = available[:5]       # fewer queries needed
    elif task == "medium":
        available = available[:7]

    # remove queries for info already fully visible
    if hospitalized is not None:
        available = [q for q in available if q != "was_patient_hospitalized"]
    if life_threatening is not None:
        available = [q for q in available if q != "was_event_life_threatening"]

    # ---- case metadata ----
    case_type = "standard"
    if interaction:
        case_type = "drug_interaction"
    elif task == "hard" and rng.random() < 0.2:
        case_type = "ambiguous"
    elif task == "hard" and rng.random() < 0.15:
        case_type = "impossible"

    # complexity score (for reward scaling)
    complexity = 1.0
    if task == "medium":
        complexity = 1.5
    elif task == "hard":
        complexity = 2.0
    if interaction:
        complexity += 0.5
    if case_type in ("ambiguous", "impossible"):
        complexity += 0.5

    # ---- build case ----
    case = {
        "observation": {
            "drug_name": drug_name,
            "symptoms": symptoms,
            "hospitalized": hospitalized,
            "life_threatening": life_threatening,
            "known_label_side_effects": known_side_effects,
            "free_text": free_text,
            "patient_age": obs_age,
            "patient_weight_kg": obs_weight,
            "concomitant_drugs": obs_concomitant,
            "medical_history": obs_history,
            "onset_days": obs_onset,
            "reporter_type": obs_reporter,
            "dechallenge_positive": obs_dechallenge,
            "rechallenge_positive": obs_rechallenge,
            "lab_values": obs_labs,
        },
        "ground_truth": {
            "severity": severity,
            "serious": serious,
            "expected": expected,
            "escalation": escalation,
            "_true_hospitalized": true_hospitalized,
            "_true_life_threatening": true_life_threatening,
        },
        "hidden_info": {},   # populated below
        "case_metadata": {
            "task": task,
            "case_type": case_type,
            "complexity": complexity,
            "drug_class": drug_info.get("class", "unknown"),
            "has_interaction": interaction is not None,
            "interaction": interaction,
            "available_queries": available,
        },
    }

    case["hidden_info"] = _build_hidden_info(case, case["ground_truth"])

    return case