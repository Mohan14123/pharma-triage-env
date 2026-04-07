---
title: PharmaTriageEnv
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Multi-Step Adverse Drug Event Triage Environment
---

# PharmaTriageEnv: Multi-Step Adverse Drug Event Triage Environment

[![Pre-Deploy Tests](https://github.com/Mohan14123/pharma-triage-env/actions/workflows/tests.yml/badge.svg)](https://github.com/Mohan14123/pharma-triage-env/actions/workflows/tests.yml)
## Overview

**PharmaTriageEnv** is a multi-step, OpenEnv-compatible environment that simulates real-world pharmacovigilance triage workflows. It evaluates an AI agent's ability to:

* Gather clinical information through targeted queries
* Interpret structured + unstructured medical data
* Classify adverse drug event (ADE) severity
* Make safety-critical escalation decisions under uncertainty

The environment models challenges drawn from the **FDA Adverse Event Reporting System (FAERS)** and ICH E2B pharmacovigilance standards, including incomplete clinical data, conflicting signals, drug interactions, and ambiguous symptom interpretation.

---

## Motivation & Real-World Alignment

### FAERS and Pharmacovigilance Workflows

The FDA FAERS database receives over **2 million adverse event reports annually**. Pharmacovigilance teams must triage each report to determine:

1. **Is this event serious?** (hospitalization, life-threatening, death, disability)
2. **Is this event expected?** (listed in the drug's approved labeling)
3. **What regulatory action is required?** (routine review vs. expedited 15-day report)

Missing critical cases leads to:
* Regulatory non-compliance (21 CFR 314.80 / ICH E2D)
* Delayed safety signal detection
* Patient harm

### How PharmaTriageEnv Approximates Real Systems

| Real-World Element | Environment Approximation |
|---|---|
| FAERS MedWatch reports | Synthetic ADE cases with structured + free-text fields |
| ICH E2B seriousness criteria | `hospitalized`, `life_threatening` boolean flags |
| Drug labeling (USPI/SmPC) | `known_label_side_effects` list |
| Pharmacovigilance officer queries | Multi-step query/reveal interaction |
| Signal detection algorithms | Deterministic grading with safety-weighted scoring |
| Drug-drug interactions | Concomitant drug lists with emergent adverse events |
| Causality assessment (Naranjo) | Dechallenge/rechallenge positive flags |
| Laboratory monitoring | Lab values (ALT, creatinine, platelets, INR) |
| Reporter credibility weighting | Reporter type (physician, nurse, patient, pharmacist) |

---

## Core Features

### Multi-Step Interaction 

Unlike single-step classification, PharmaTriageEnv supports a **query/reveal loop**:

```
Agent: receives initial (partial) observation
Agent: asks "was_patient_hospitalized?"        → Env reveals: "Records confirm patient WAS hospitalized."
Agent: asks "what_are_lab_values?"             → Env reveals: "ALT: 520, creatinine: 4.2..."
Agent: submits final triage decision           → Env grades and returns reward
```

Available queries (10 total):
* `was_patient_hospitalized` — confirm hospitalization status
* `was_event_life_threatening` — confirm life-threatening status
* `what_are_concomitant_drugs` — reveal drug interaction risk
* `what_is_medical_history` — reveal comorbidities
* `was_dechallenge_positive` — did stopping the drug help?
* `was_rechallenge_positive` — did restarting reproduce the event?
* `what_are_lab_values` — reveal ALT, creatinine, hemoglobin, platelets, INR
* `what_is_reporter_type` — physician/nurse/patient/pharmacist
* `what_is_onset_timeline` — days between drug start and event
* `is_symptom_in_drug_label` — check if symptoms are listed side effects

### Decision Output

```json
{
  "severity": "low | medium | high",
  "serious": true | false,
  "expected": true | false,
  "escalation": "routine_review | urgent_review | regulatory_report"
}
```

---

## Task Difficulty Levels

| Task | Description | Difficulty | Max Steps |
|------|-------------|------------|-----------|
| Easy | Complete clean data, aligned signals, full side-effect list | Low | 3 |
| Medium | Partial observability, noisy inputs, some ambiguity, abbreviations | Moderate | 5 |
| Hard | Adversarial cases, drug interactions, OCR errors, contradictory signals, escalation traps | High | 5 |

### Expected Baseline Performance

| Task | Expected Score | Reward Range |
|------|---------------|--------------|
| Easy | 0.98 – 1.0 | +4 to +6 |
| Medium | 0.90 – 0.97 | +5 to +10 |
| Hard | 0.55 – 0.75 | -4 to +10 |

---

## Environment Design

### Observation Space

```json
{
  "drug_name": "Warfarin",
  "symptoms": ["GI bleeding", "bruising"],
  "hospitalized": null,
  "life_threatening": null,
  "known_label_side_effects": ["bruising", "bleeding gums"],
  "free_text": "Pt (72y, 85kg) reports GI bleeding after taking Warfarin. Also taking Ibuprofen. Hx: atrial fibrillation. Report received via fax — some fields unclear.",
  "patient_age": 72,
  "patient_weight_kg": 85.0,
  "concomitant_drugs": ["Ibuprofen"],
  "medical_history": ["atrial fibrillation"],
  "onset_days": null,
  "reporter_type": "pharmacist",
  "dechallenge_positive": null,
  "rechallenge_positive": null,
  "lab_values": null,
  "step_number": 0,
  "available_queries": ["was_patient_hospitalized", "was_event_life_threatening", "what_are_lab_values", "..."],
  "query_response": null,
  "max_steps": 5
}
```

Key characteristics:
* **Partial observability** — fields may be null, requiring queries
* **Structured + unstructured inputs** — clinical flags + free text
* **Real-world artifacts** — typos, OCR errors, medical abbreviations
* **Drug interactions** — concomitant drugs with emergent ADE risk
* **Contradictory signals** — free text may conflict with structured data

### Drug Database (12 drugs, FAERS-inspired)

| Drug | Class | Notable Rare ADEs |
|------|-------|-------------------|
| Metformin | Antidiabetic | Lactic acidosis |
| Warfarin | Anticoagulant | Hemorrhagic stroke, skin necrosis |
| Amoxicillin | Antibiotic | Anaphylaxis, Stevens-Johnson syndrome |
| Ciprofloxacin | Fluoroquinolone | Tendon rupture, aortic dissection |
| Carbamazepine | Anticonvulsant | Aplastic anemia, DRESS syndrome |
| Methotrexate | Immunosuppressant | Pancytopenia, pulmonary fibrosis |
| Sertraline | SSRI | Serotonin syndrome |
| ... | ... | ... |

### Known Drug Interactions

| Drug Pair | Emergent ADE | Severity |
|-----------|-------------|----------|
| Warfarin + Ibuprofen | GI bleeding | High |
| Sertraline + Methotrexate | Serotonin syndrome | High |
| Lisinopril + Ibuprofen | Renal failure | High |
| Omeprazole + Methotrexate | Methotrexate toxicity | High |
| Ciprofloxacin + Warfarin | Hemorrhagic event | High |

---

## Reward vs Score

### Reward (Shaped Learning Signal)

* Dynamic, component-based with partial credit
* Range: **-10 to +10**
* Components:
  * Base correctness (severity, seriousness, expectedness, escalation)
  * Safety-critical bonuses (+3 for catching novel ADEs)
  * Severe penalties (-4 for missing serious cases, -2 for missing novel)
  * Query efficiency rewards (+0.5 for ≤3 queries)
  * Step looping penalties
  * Invalid action penalties
  * Complexity scaling by task difficulty

### Score (Final Evaluation Metric)

* Deterministic grader output
* Normalized to **0.0 – 1.0**
* Per-task weight profiles
* Safety-compound check (serious + unexpected → regulatory_report)
* Partial credit for adjacent severity/escalation levels

---

## Architecture

```
pharma-triage-env/
├── src/pharma_triage_env/
│   ├── __init__.py          # Package exports
│   ├── env.py               # Multi-step environment (reset/step/state)
│   ├── generator.py         # Complex case generator (12 drugs, interactions, artifacts)
│   ├── grader.py             # Deterministic grader (score 0-1)
│   ├── reward.py             # Shaped reward calculator (-10 to +10)
│   ├── models.py             # Pydantic models (Observation, Action, Reward)
│   ├── tasks.py              # Task definitions (easy/medium/hard)
│
├── inference.py              # OpenAI-client inference (env vars, fallback, logging)
├── scripts/
│   ├── demo.py               # Baseline agent demo
│   └── smoke_test.py         # Automated validation
├── server/
│   └── app.py                # FastAPI server (multi-step endpoints)
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                # Docker deployment
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run demo (baseline agent)

```bash
PYTHONPATH=./src python scripts/demo.py
```

### 3. Run smoke tests

```bash
PYTHONPATH=./src python scripts/smoke_test.py
```

### 4. Run inference (requires LLM API)

```bash
export HF_TOKEN=<your_token>
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
PYTHONPATH=./src python inference.py
```

### 5. Run server

```bash
PYTHONPATH=./src python server/app.py
```

### Demo Output — Full Explanation

Run `PYTHONPATH=./src python scripts/demo.py` to see output like this:

```
==================================================
  TASK: EASY
==================================================
  ✓ Ep 01 | Score: 0.900 | Reward:  +4.00 | Steps: 1 | Queries: 0 | Type: standard
  ✓ Ep 02 | Score: 1.000 | Reward:  +5.00 | Steps: 1 | Queries: 0 | Type: standard
  ...
  ✗ Ep 01 | Score: 0.076 | Reward:  -3.30 | Steps: 5 | Queries: 4 | Type: impossible
  △ Ep 10 | Score: 0.441 | Reward:  +4.06 | Steps: 1 | Queries: 0 | Type: ambiguous

  --- SUMMARY (HARD) ---
  Average Score  : 0.5817
  Average Reward : +4.357
  Avg Queries    : 1.2
  Avg Steps      : 2.3
  Min/Max Score  : 0.076 / 1.000
```

---

#### Episode Line Format

```
  <status> Ep <##> | Score: <X.XXX> | Reward: <±XX.XX> | Steps: <N> | Queries: <N> | Type: <type>
```

| Field | Meaning |
|---|---|
| **Status icon** | `✓` = score ≥ 0.8 (pass), `△` = 0.4–0.79 (partial), `✗` = < 0.4 (fail) |
| **Ep ##** | Episode number within the current task (1-indexed) |
| **Score** | Final deterministic grade from `grader.py`, normalized to `[0.0, 1.0]` |
| **Reward** | Shaped learning signal from `reward.py`, range `[-10, +10]` |
| **Steps** | Total environment steps taken (queries + final decision action) |
| **Queries** | Number of information-reveal queries issued before the decision |
| **Type** | The case category (see below) |

---

#### Case Types

| Type | What it means | Can score 1.0? |
|---|---|---|
| `standard` | Normal ADE case with complete or partially-observable data | Yes |
| `drug_interaction` | Patient is on multiple drugs with a known emergent interaction | Yes (requires detecting interaction) |
| `ambiguous` | `expected` field is deliberately flipped; drug label signal conflicts with symptoms | Rarely — ambiguity lowers ceiling |
| `impossible` | Ground truth is structurally contradicted: `serious` flipped, observable flags inverted, escalation trapped | No — 1.0 is mathematically unreachable |

> **Why can't `impossible` cases score 1.0?**  
> The environment injects three simultaneous contradictions into the case: the `hospitalized`/`life_threatening` flags shown to the agent are the **opposite** of the ground truth `serious` label, the `escalation` target is always flipped to a wrong value, and the `serious` ground truth itself is inverted. No consistent answer to all five grading dimensions exists.

---

#### Score Math

The score is a weighted sum across five grading dimensions, normalized by a per-task maximum:

| Dimension | EASY weight | MEDIUM weight | HARD weight |
|---|---|---|---|
| Severity (exact or ±1 partial) | 1.5 | 2.0 | 2.0 |
| Seriousness (exact match) | 2.5 | 3.0 | 3.5 |
| Expectedness (exact match) | 1.0 | 1.5 | 2.0 |
| Escalation (exact or ±1 partial) | 2.0 | 2.5 | 3.0 |
| Safety compound check | 2.0 | 3.0 | 4.0 |
| **Max possible raw score** | **9.0** | **12.0** | **14.5** |

**Safety compound check rules:**
- If case is **critical** (`serious=True` AND `expected=False`): agent must correctly get all three (`serious`, `expected`, `escalation=regulatory_report`) — scored proportionally (0→3 checks passed)
- If case is **non-critical**: agent earns full compound weight for correctly **not** over-escalating to `regulatory_report`; zero if it falsely escalates

Common score values and their cause:

| Score | Typical cause |
|---|---|
| `1.000` | All 5 dimensions correct |
| `0.900` | Severity off by one step (partial credit = `1.5 × 0.4 = 0.6` instead of `1.5`) |
| `0.855` | Escalation off by one step (partial credit = `escalation_weight × 0.3`) |
| `0.678` | Ambiguous case — `expected` signal deliberately flipped, one dimension unavoidably wrong |
| `0.441 / 0.533` | Multiple dimensions wrong (typical hard case miss) |
| `0.276` | Impossible case — 2 of 5 dimensions are structurally unresolvable |
| `0.076` | Impossible case — 3+ dimensions contradicted, penalties applied |
| `0.000` | Catastrophic failure — missed serious case + wrong escalation + penalties exceed score |

---

#### Summary Block

```
  --- SUMMARY (HARD) ---
  Average Score  : 0.5817    ← mean grade across all 25 episodes
  Average Reward : +4.357    ← mean shaped reward
  Avg Queries    : 1.2       ← how many info-lookups per episode on average
  Avg Steps      : 2.3       ← total env steps per episode (queries + decision)
  Min/Max Score  : 0.076 / 1.000   ← best and worst episode score
```

---

#### Reproducibility

All episodes use a fixed, deterministic seed chain:

```python
SEED_BASE = 42  # in demo.py
# Episode N gets seed: 42 + N
# Same seed → identical case every run
```

To benchmark against a different case set, change `SEED_BASE`. All reproducibility guarantees hold for any integer seed.

---

## Inference Script Compliance

The `inference.py` script strictly follows OpenEnv hackathon requirements:

* ✅ `from openai import OpenAI`
* ✅ `OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)`
* ✅ Environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
* ✅ `HF_TOKEN` has **NO default value**
* ✅ `temperature=0`, `max_tokens=256` for deterministic output
* ✅ Step loop with `MAX_STEPS` limit
* ✅ Deterministic fallback action
* ✅ Structured logging format:

```
START task=<task_name>
STEP action=<action> reward=<reward>
END score=<final_score>
```

---

## OpenEnv Compliance

* ✅ Typed Pydantic models: `Observation`, `Action`, `Reward`
* ✅ API: `reset() → Observation`, `step(action) → (Observation, Reward, done, info)`, `state() → dict`
* ✅ `openenv.yaml` manifest with task definitions
* ✅ Deterministic grading system
* ✅ 3 difficulty tiers (easy, medium, hard)
* ✅ Score ∈ [0.0, 1.0], Reward ∈ [-10.0, +10.0]

---

## Deployment

### Environment Variables

```
API_BASE_URL    # default: https://router.huggingface.co/v1
MODEL_NAME      # default: Qwen/Qwen2.5-72B-Instruct
HF_TOKEN        # required, no default
```

### Docker

```bash
docker build -t pharma-env .
docker run -p 7860:7860 -e HF_TOKEN=<token> pharma-env
```

### Hugging Face Spaces

The Dockerfile is configured for HF Spaces deployment (port 7860).

---

## Evaluation Alignment

| Criterion | Implementation |
|-----------|---------------|
| **Real-world utility** | FAERS-inspired pharmacovigilance domain |
| **Task quality** | Deterministic grading + 3-tier difficulty scaling |
| **Environment design** | Multi-step interaction, drug interactions, partial observability, adversarial noise |
| **Code quality** | Modular, typed, reproducible, spec-compliant |
| **Creativity** | Query/reveal mechanic, drug interaction detection, real-world data artifacts |

---

## Limitations

* Simplified clinical reasoning rules (not a full Naranjo assessment)
* Synthetic data (not real FAERS reports)
* Limited drug database (12 drugs)
* No temporal event tracking across multiple reports

---

## Future Work

* Integration with real FAERS data (anonymized)
* Full Naranjo causality assessment scoring
* Multi-report signal detection (aggregate analysis)
* Temporal event tracking
* Expanded drug interaction database
* Integration with MedDRA coding

---

## Authors

* Mohan Kumar S
* Ashwin R
* Manikandan S

---

## License

MIT
