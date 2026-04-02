import random

DRUGS = ["DrugA", "DrugB", "DrugC"]

SYMPTOMS = [
    "rash", "fever", "headache",
    "anaphylaxis", "cardiac arrest", "nausea"
]

SEVERE_SYMPTOMS = ["anaphylaxis", "cardiac arrest"]

NOISE_TEXT = [
    "Patient unsure about symptoms",
    "Symptoms appeared gradually",
    "Initial mild reaction but worsened later",
    "Patient reports unclear timeline",
    "Symptoms may be unrelated",
    "Symptoms seem mild overall",
    "No major concerns reported initially"
]


def maybe_missing(value, prob=0.25):
    return None if random.random() < prob else value


def generate_case(task="hard"):
    drug = random.choice(DRUGS)

    # ========================
    # SYMPTOMS
    # ========================
    num_symptoms = random.randint(1, 2)
    symptoms = random.sample(SYMPTOMS, k=num_symptoms)

    if task == "hard" and random.random() < 0.5:
        symptoms.append(random.choice(SEVERE_SYMPTOMS))

    # ========================
    # CLINICAL FLAGS
    # ========================
    hospitalized = random.choice([True, False])
    life_threatening = random.choice([True, False])

    if task == "hard" and random.random() < 0.3:
        hospitalized = False
        life_threatening = True

    # ========================
    # KNOWN SIDE EFFECTS
    # ========================
    if task == "easy":
        known_side_effects = symptoms[:]

        # ✅ Slight realism noise (20%)
        if random.random() < 0.2:
            known_side_effects = []

    else:
        known_side_effects = random.sample(SYMPTOMS, k=random.randint(1, 2))

        if task in ["medium", "hard"] and random.random() < 0.3:
            known_side_effects = []

    # ========================
    # PARTIAL OBSERVABILITY
    # ========================
    if task == "medium":
        hospitalized = maybe_missing(hospitalized, 0.2)
        life_threatening = maybe_missing(life_threatening, 0.2)

    if task == "hard":
        hospitalized = maybe_missing(hospitalized, 0.3)
        life_threatening = maybe_missing(life_threatening, 0.3)

        if random.random() < 0.3:
            known_side_effects = None

    # ========================
    # GROUND TRUTH
    # ========================
    serious = bool(hospitalized) or bool(life_threatening)

    # expectedness
    if known_side_effects:
        expected = any(s in known_side_effects for s in symptoms)
    else:
        expected = False

    # HARD: deceptive flip
    if task == "hard" and random.random() < 0.4:
        expected = not expected

    # severity
    if task == "easy":
        severity = "high" if serious else "medium"
    else:
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

    # HARD: escalation trap
    if task == "hard" and random.random() < 0.3:
        escalation = random.choice([
            "routine_review",
            "urgent_review",
            "regulatory_report"
        ])

    # ========================
    # FREE TEXT
    # ========================
    base_text = f"Patient reports {', '.join(symptoms)} after taking {drug}."

    if task == "easy":
        free_text = base_text

    elif task == "medium":
        free_text = f"{base_text} {random.choice(NOISE_TEXT)}"

    else:
        free_text = f"{base_text} {random.choice(NOISE_TEXT)}"

        if random.random() < 0.4:
            free_text += " Symptoms appear mild despite complications."

    # ========================
    # FINAL CASE
    # ========================
    return {
        "observation": {
            "drug_name": drug,
            "symptoms": symptoms,
            "hospitalized": hospitalized,
            "life_threatening": life_threatening,
            "known_label_side_effects": known_side_effects,
            "free_text": free_text
        },
        "ground_truth": {
            "severity": severity,
            "serious": serious,
            "expected": expected,
            "escalation": escalation
        }
    }