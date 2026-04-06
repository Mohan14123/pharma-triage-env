"""
Inference Script — PharmaTriageEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=hard env=pharma-triage-env model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=query:was_patient_hospitalized reward=0.20 done=false error=null
    [STEP] step=2 action=query:what_are_lab_values reward=0.20 done=false error=null
    [STEP] step=3 action=decide(high,true,false,regulatory_report) reward=7.50 done=true error=null
    [END] success=true steps=3 score=0.920 rewards=0.20,0.20,7.50
"""

import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Action

# ============================================================
# ENV VARS (strict compliance — HF_TOKEN raises error if missing)
# ============================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required but not set.")
BENCHMARK = "pharma-triage-env"

# ============================================================
# CONSTANTS
# ============================================================
MAX_STEPS = 5
TEMPERATURE = 0
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5
NUM_EPISODES = 30

# Reproducibility seed: ensures identical cases across runs.
# Each task gets a separate seed range (spaced by NUM_EPISODES)
# to avoid overlap: easy=42..71, medium=72..101, hard=102..131
SEED = 42

# ============================================================
# OPENAI CLIENT
# ============================================================
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = textwrap.dedent("""
You are a pharmacovigilance triage agent analyzing adverse drug event (ADE) reports.

You operate in a multi-step loop. Each turn you MUST return ONLY valid JSON (no explanation).

OPTION A — Ask for more information (recommended when critical fields are missing):
{"query": "<query_name>"}

Available queries:
  was_patient_hospitalized, was_event_life_threatening, what_are_concomitant_drugs,
  what_is_medical_history, was_dechallenge_positive, was_rechallenge_positive,
  what_are_lab_values, what_is_reporter_type, what_is_onset_timeline,
  is_symptom_in_drug_label

OPTION B — Submit your final triage decision:
{
  "severity": "low" | "medium" | "high",
  "serious": true | false,
  "expected": true | false,
  "escalation": "routine_review" | "urgent_review" | "regulatory_report"
}

Clinical rules:
- serious = patient was hospitalized OR event was life-threatening
- expected = reported symptoms appear in the drug's known side-effect label
- If serious AND unexpected → escalation = "regulatory_report"
- If serious AND expected → escalation = "urgent_review"
- If not serious → escalation = "routine_review"
- severity "high" = severe/life-threatening symptoms (anaphylaxis, cardiac arrest, etc.)
- severity "medium" = moderate clinical concern
- severity "low" = mild, self-limiting

Consider: drug interactions, abnormal lab values, patient history, reporter credibility.
Query for missing critical info (hospitalization, life-threatening status) before deciding.
Max 3 queries recommended. After that, decide with available information.

Return ONLY valid JSON. No markdown. No explanations.
""").strip()


# ============================================================
# STRUCTURED LOGGING (exact format compliance)
# ============================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================
# MODEL CALL
# ============================================================
def call_model(obs_data: dict, conversation_history: list) -> Optional[dict]:
    """Call the LLM with observation context. Returns parsed dict or None on failure."""
    # build user prompt with clinical context emphasis
    obs_summary = _build_clinical_summary(obs_data)

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": obs_summary})

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        content = (completion.choices[0].message.content or "").strip()

        # strip markdown code fences if model wraps output
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content[3:-3]
            content = content.strip()

        return json.loads(content)
    except Exception:
        return None


def _build_clinical_summary(obs_data: dict) -> str:
    """Build a structured clinical summary from observation data for the LLM."""
    parts = [f"=== ADE Report (Step {obs_data.get('step_number', 0)}/{obs_data.get('max_steps', 5)}) ==="]

    parts.append(f"Drug: {obs_data.get('drug_name', 'unknown')}")
    parts.append(f"Symptoms: {', '.join(obs_data.get('symptoms', []))}")

    # structured fields (may be null)
    hosp = obs_data.get("hospitalized")
    lt = obs_data.get("life_threatening")
    parts.append(f"Hospitalized: {hosp if hosp is not None else 'UNKNOWN — consider querying'}")
    parts.append(f"Life-threatening: {lt if lt is not None else 'UNKNOWN — consider querying'}")

    kse = obs_data.get("known_label_side_effects")
    if kse is not None:
        parts.append(f"Known side effects: {', '.join(kse) if kse else 'none listed'}")
    else:
        parts.append("Known side effects: UNKNOWN — consider querying")

    # rich clinical fields
    if obs_data.get("patient_age") is not None:
        parts.append(f"Patient age: {obs_data['patient_age']}y")
    if obs_data.get("patient_weight_kg") is not None:
        parts.append(f"Patient weight: {obs_data['patient_weight_kg']}kg")
    if obs_data.get("concomitant_drugs"):
        parts.append(f"Concomitant drugs: {', '.join(obs_data['concomitant_drugs'])}")
    if obs_data.get("medical_history"):
        parts.append(f"Medical history: {', '.join(obs_data['medical_history'])}")
    if obs_data.get("onset_days") is not None:
        parts.append(f"Onset: {obs_data['onset_days']} days after starting drug")
    if obs_data.get("reporter_type"):
        parts.append(f"Reporter: {obs_data['reporter_type']}")
    if obs_data.get("dechallenge_positive") is not None:
        parts.append(f"Dechallenge positive: {obs_data['dechallenge_positive']}")
    if obs_data.get("rechallenge_positive") is not None:
        parts.append(f"Rechallenge positive: {obs_data['rechallenge_positive']}")
    if obs_data.get("lab_values"):
        labs = obs_data["lab_values"]
        lab_str = ", ".join(f"{k}={v}" for k, v in labs.items())
        parts.append(f"Lab values: {lab_str}")

    # free text
    if obs_data.get("free_text"):
        parts.append(f"\nFree text narrative:\n{obs_data['free_text']}")

    # query response from previous step
    if obs_data.get("query_response"):
        parts.append(f"\n>> Query response: {obs_data['query_response']}")

    # available queries
    available = obs_data.get("available_queries", [])
    if available:
        parts.append(f"\nAvailable queries: {', '.join(available)}")

    return "\n".join(parts)


# ============================================================
# FALLBACK ACTION (deterministic — no randomness, no external calls)
# ============================================================
def fallback(obs_data: dict) -> dict:
    """Deterministic rule-based fallback agent."""
    hospitalized = obs_data.get("hospitalized")
    life_threatening = obs_data.get("life_threatening")
    symptoms = obs_data.get("symptoms", [])
    known = obs_data.get("known_label_side_effects") or []

    serious = bool(hospitalized) or bool(life_threatening)
    expected = bool(set(symptoms) & set(known))

    # detect severe symptoms by keyword matching
    severe_keywords = [
        "anaphylaxis", "cardiac arrest", "hemorrhagic", "rhabdomyolysis",
        "Stevens-Johnson", "serotonin syndrome", "aplastic anemia",
        "pulmonary fibrosis", "pancytopenia", "aortic dissection",
        "GI bleeding", "renal failure", "hepatotoxicity", "skin necrosis",
        "DRESS syndrome", "QT prolongation", "tendon rupture",
    ]
    has_severe = any(
        kw.lower() in s.lower() for s in symptoms for kw in severe_keywords
    )

    # check lab values for critical abnormalities
    labs = obs_data.get("lab_values") or {}
    if labs.get("ALT", 0) > 200:
        has_severe = True
    if labs.get("creatinine", 0) > 3.0:
        has_severe = True
    if labs.get("platelets", 999) < 50:
        has_severe = True
    if labs.get("INR", 0) > 4.0:
        has_severe = True

    # check concomitant drugs for known dangerous pairs
    concomitant = obs_data.get("concomitant_drugs") or []
    drug = obs_data.get("drug_name", "")
    dangerous_pairs = {
        frozenset(["Warfarin", "Ibuprofen"]),
        frozenset(["Sertraline", "Methotrexate"]),
        frozenset(["Lisinopril", "Ibuprofen"]),
        frozenset(["Ciprofloxacin", "Warfarin"]),
        frozenset(["Omeprazole", "Methotrexate"]),
    }
    all_drugs = {drug} | set(concomitant)
    for pair in dangerous_pairs:
        if pair.issubset(all_drugs):
            has_severe = True
            break

    if has_severe:
        severity = "high"
        serious = True
    elif serious:
        severity = "medium"
    else:
        severity = "low"

    if serious and not expected:
        escalation = "regulatory_report"
    elif serious:
        escalation = "urgent_review"
    else:
        escalation = "routine_review"

    return {
        "severity": severity,
        "serious": serious,
        "expected": expected,
        "escalation": escalation,
    }


# ============================================================
# QUERY STRATEGY (smart info-gathering)
# ============================================================
def _should_query(obs_data: dict, step: int) -> Optional[str]:
    """Determine if the agent should query for missing info before deciding.
    Returns query name or None if ready to decide."""
    if step >= MAX_STEPS - 1:
        return None  # must decide on last step

    available = obs_data.get("available_queries") or []

    # priority 1: hospitalization status is critical for seriousness
    if obs_data.get("hospitalized") is None and "was_patient_hospitalized" in available:
        return "was_patient_hospitalized"

    # priority 2: life-threatening status
    if obs_data.get("life_threatening") is None and "was_event_life_threatening" in available:
        return "was_event_life_threatening"

    # priority 3: side effect label (needed for expectedness)
    if obs_data.get("known_label_side_effects") is None and "is_symptom_in_drug_label" in available:
        return "is_symptom_in_drug_label"

    # priority 4: lab values if symptoms suggest organ damage
    symptoms_lower = [s.lower() for s in obs_data.get("symptoms", [])]
    organ_keywords = ["hepatotoxicity", "renal", "bleeding", "pancytopenia", "rhabdomyolysis"]
    if obs_data.get("lab_values") is None and "what_are_lab_values" in available:
        if any(kw in " ".join(symptoms_lower) for kw in organ_keywords):
            return "what_are_lab_values"

    return None


# ============================================================
# RUN SINGLE EPISODE
# ============================================================
def run_episode(env: PharmaTriageEnv, task: str) -> tuple:
    """Run a single episode. Returns (score, rewards_list, steps_taken, success)."""
    obs = env.reset()
    obs_data = obs.model_dump()
    conversation_history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            # try LLM first, use query strategy as enhancement
            result = call_model(obs_data, conversation_history)

            if result is None:
                # LLM failed — use smart query strategy or fallback
                smart_query = _should_query(obs_data, step)
                if smart_query:
                    result = {"query": smart_query}
                else:
                    result = fallback(obs_data)

            # --- QUERY action ---
            is_query = "query" in result and result.get("query") and step < MAX_STEPS
            if is_query:
                action = Action(query=result["query"])
                obs, reward, done, info = env.step(action)
                obs_data = obs.model_dump()
                reward_val = reward.value

                action_str = f"query:{result['query']}"
                error_msg = info.get("error")
                rewards.append(reward_val)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_msg)

                # update conversation context
                conversation_history.append(
                    {"role": "assistant", "content": json.dumps(result)}
                )
                conversation_history.append(
                    {"role": "user", "content": _build_clinical_summary(obs_data)}
                )

                if done:
                    score = info.get("score", 0.0)
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    break

            # --- DECISION action ---
            else:
                action = Action(
                    severity=result.get("severity", "medium"),
                    serious=result.get("serious", False),
                    expected=result.get("expected", False),
                    escalation=result.get("escalation", "routine_review"),
                )
                obs, reward, done, info = env.step(action)
                reward_val = reward.value

                sev = result.get("severity", "?")
                ser = result.get("serious", "?")
                exp = result.get("expected", "?")
                esc = result.get("escalation", "?")
                action_str = f"decide({sev},{ser},{exp},{esc})"

                error_msg = info.get("error")
                rewards.append(reward_val)
                steps_taken = step
                score = info.get("score", 0.0)
                success = score >= SUCCESS_SCORE_THRESHOLD

                log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_msg)
                break
        else:
            # max steps exceeded — force fallback decision
            fallback_result = fallback(obs_data)
            action = Action(**fallback_result)
            obs, reward, done, info = env.step(action)
            reward_val = reward.value

            action_str = "fallback(forced)"
            rewards.append(reward_val)
            steps_taken = MAX_STEPS
            score = info.get("score", 0.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

            log_step(step=MAX_STEPS, action=action_str, reward=reward_val, done=done, error=None)

    except Exception as exc:
        error_msg = str(exc)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=error_msg)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, rewards, steps_taken, success


# ============================================================
# MAIN
# ============================================================
def main():
    tasks = ["easy", "medium", "hard"]
    overall_scores = {}

    for idx, task in enumerate(tasks):
        task_seed = SEED + (idx * NUM_EPISODES)
        env = PharmaTriageEnv(task=task, max_steps=MAX_STEPS, seed=task_seed)
        task_scores = []

        for _ in range(NUM_EPISODES):
            score, _, _, _ = run_episode(env, task)
            task_scores.append(score)

        avg_score = sum(task_scores) / len(task_scores)
        overall_scores[task] = avg_score

    # final summary
    overall = sum(overall_scores.values()) / len(overall_scores)
    all_rewards = [overall]
    log_start(task="summary", env=BENCHMARK, model=MODEL_NAME)
    log_step(step=1, action="aggregate", reward=overall, done=True, error=None)
    log_end(success=overall >= SUCCESS_SCORE_THRESHOLD, steps=1, score=overall, rewards=all_rewards)


if __name__ == "__main__":
    main()
