"""
Demo script for PharmaTriageEnv.

Runs a deterministic baseline agent across all 3 task difficulties,
demonstrating the multi-step interaction loop.

Uses fixed seeds for full reproducibility.

Usage:
    PYTHONPATH=./src python scripts/demo.py
"""

from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Action


# ============================================================
# SEVERE SYMPTOM KEYWORDS
# ============================================================
SEVERE_KEYWORDS = [
    "anaphylaxis", "cardiac arrest", "hemorrhagic", "rhabdomyolysis",
    "stevens-johnson", "serotonin syndrome", "aplastic anemia",
    "pulmonary fibrosis", "pancytopenia", "aortic dissection",
    "gi bleeding", "renal failure", "hepatotoxicity", "skin necrosis",
    "dress syndrome", "qt prolongation", "tendon rupture",
    "hemorrhagic stroke", "methotrexate toxicity",
]

# Known dangerous drug pairs
DANGEROUS_PAIRS = [
    {"Warfarin", "Ibuprofen"},
    {"Sertraline", "Methotrexate"},
    {"Lisinopril", "Ibuprofen"},
    {"Ciprofloxacin", "Warfarin"},
    {"Omeprazole", "Methotrexate"},
    {"Amlodipine", "Atorvastatin"},
    {"Carbamazepine", "Warfarin"},
]


def _has_severe_symptom(symptoms):
    """Check if any symptom matches a severe keyword."""
    for s in symptoms:
        s_lower = s.lower()
        for kw in SEVERE_KEYWORDS:
            if kw in s_lower:
                return True
    return False


def _has_dangerous_interaction(drug_name, concomitant):
    """Check if current drugs form a known dangerous pair."""
    if not concomitant:
        return False
    all_drugs = {drug_name} | set(concomitant)
    for pair in DANGEROUS_PAIRS:
        if pair.issubset(all_drugs):
            return True
    return False


def _check_labs_critical(labs):
    """Check lab values for critical abnormalities."""
    if not labs:
        return False
    if labs.get("ALT", 0) > 200:
        return True
    if labs.get("creatinine", 0) > 3.0:
        return True
    if labs.get("platelets", 999) < 50:
        return True
    if labs.get("INR", 0) > 4.0:
        return True
    if labs.get("hemoglobin", 99) < 7.0:
        return True
    if labs.get("WBC", 99) < 2.5:
        return True
    return False


def _parse_free_text_signals(free_text):
    """Extract clinical signals from free text."""
    if not free_text:
        return {"hosp_hint": None, "lt_hint": None, "severe_hint": False}

    text_lower = free_text.lower()

    hosp_hint = None
    if any(kw in text_lower for kw in ["admitted", "hospitalized", "hospitalised",
                                        "icu", "emergency room", "inpatient",
                                        "discharge summary"]):
        hosp_hint = True
    elif any(kw in text_lower for kw in ["not hospitalized", "outpatient", "sent home"]):
        hosp_hint = False

    lt_hint = None
    if any(kw in text_lower for kw in ["life-threatening", "life threatening",
                                        "critical", "resuscitated", "cardiac arrest",
                                        "icu admission"]):
        lt_hint = True

    severe_hint = any(kw in text_lower for kw in SEVERE_KEYWORDS)

    return {"hosp_hint": hosp_hint, "lt_hint": lt_hint, "severe_hint": severe_hint}


def baseline_agent(obs, step_num=0):
    """
    Deterministic baseline agent with multi-step interaction.

    Strategy:
    - Step 0: query hospitalization if missing
    - Step 1: query life-threatening if missing
    - Step 2: query side-effect label if missing
    - Step 3: query labs if symptoms suggest organ damage
    - Then: make decision using ALL available signals (structured + free text + labs)
    """
    available = obs.available_queries or []

    # === QUERY PHASE: gather critical missing info ===

    # Priority 1: hospitalization status
    if step_num <= 1 and obs.hospitalized is None and "was_patient_hospitalized" in available:
        return Action(query="was_patient_hospitalized")

    # Priority 2: life-threatening status
    if step_num <= 2 and obs.life_threatening is None and "was_event_life_threatening" in available:
        return Action(query="was_event_life_threatening")

    # Priority 3: side effect label (needed for expectedness)
    if step_num <= 2 and obs.known_label_side_effects is None and "is_symptom_in_drug_label" in available:
        return Action(query="is_symptom_in_drug_label")

    # Priority 4: lab values if symptoms hint at organ damage
    symptoms = obs.symptoms or []
    symptoms_lower = [s.lower() for s in symptoms]
    organ_hints = ["hepatotoxicity", "renal", "bleeding", "pancytopenia",
                   "rhabdomyolysis", "gi bleeding", "aplastic"]
    if step_num <= 3 and obs.lab_values is None and "what_are_lab_values" in available:
        if any(h in " ".join(symptoms_lower) for h in organ_hints):
            return Action(query="what_are_lab_values")

    # === DECISION PHASE: use ALL available signals ===

    known = obs.known_label_side_effects or []
    concomitant = obs.concomitant_drugs or []

    # -- Seriousness: combine structured flags, free text, labs, interactions --
    serious = bool(obs.hospitalized) or bool(obs.life_threatening)

    # Parse free text for additional signals
    ft_signals = _parse_free_text_signals(obs.free_text)
    if ft_signals["hosp_hint"] is True and obs.hospitalized is None:
        serious = True
    if ft_signals["lt_hint"] is True:
        serious = True

    # Check severe symptoms
    has_severe = _has_severe_symptom(symptoms) or ft_signals["severe_hint"]

    # Check drug interactions
    has_interaction = _has_dangerous_interaction(obs.drug_name, concomitant)

    # Check lab values
    labs_critical = _check_labs_critical(obs.lab_values)

    # Override seriousness if severe indicators found
    if has_severe or has_interaction or labs_critical:
        serious = True

    # -- Expectedness --
    expected = any(s in known for s in symptoms)

    # -- Severity --
    if has_severe or labs_critical:
        severity = "high"
    elif serious:
        severity = "medium"
    else:
        severity = "low"

    # Boost severity for drug interactions
    if has_interaction and severity != "high":
        severity = "high"

    # -- Escalation --
    if serious and not expected:
        escalation = "regulatory_report"
    elif serious:
        escalation = "urgent_review"
    else:
        escalation = "routine_review"

    return Action(
        severity=severity,
        serious=serious,
        expected=expected,
        escalation=escalation,
    )


def run_demo():
    tasks = ["easy", "medium", "hard"]
    SEED_BASE = 42  # fixed seed for reproducibility

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*50}")

        env = PharmaTriageEnv(task=task, max_steps=5, seed=SEED_BASE)

        scores = []
        rewards = []
        total_queries = 0
        total_steps = 0

        num_episodes = 25

        for i in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0

            for step_num in range(env.max_steps):
                action = baseline_agent(obs, step_num)

                obs, reward, done, info = env.step(action)
                reward_val = reward.value if hasattr(reward, "value") else 0.0
                episode_reward += reward_val

                if done:
                    score = info.get("score", 0.0)
                    queries = info.get("queries_used", 0)
                    steps = info.get("steps_taken", step_num + 1)
                    case_type = info.get("case_type", "standard")

                    scores.append(score)
                    rewards.append(episode_reward)
                    total_queries += queries
                    total_steps += steps

                    status = "✓" if score >= 0.7 else "△" if score >= 0.4 else "✗"
                    print(
                        f"  {status} Ep {i+1:02d} | "
                        f"Score: {score:.3f} | "
                        f"Reward: {episode_reward:+6.2f} | "
                        f"Steps: {steps} | "
                        f"Queries: {queries} | "
                        f"Type: {case_type}"
                    )
                    break

        avg_score = sum(scores) / len(scores)
        avg_reward = sum(rewards) / len(rewards)
        avg_queries = total_queries / num_episodes
        avg_steps = total_steps / num_episodes

        print(f"\n  --- SUMMARY ({task.upper()}) ---")
        print(f"  Average Score  : {avg_score:.4f}")
        print(f"  Average Reward : {avg_reward:+.3f}")
        print(f"  Avg Queries    : {avg_queries:.1f}")
        print(f"  Avg Steps      : {avg_steps:.1f}")
        print(f"  Min/Max Score  : {min(scores):.3f} / {max(scores):.3f}")

    print(f"\n{'='*50}")
    print("  Demo completed successfully.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_demo()