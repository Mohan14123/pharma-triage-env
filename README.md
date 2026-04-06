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

### Multi-Step Interaction (Key Differentiator)

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
| Easy | 0.85 – 1.0 | +3 to +10 |
| Medium | 0.60 – 0.85 | +1 to +7 |
| Hard | 0.40 – 0.70 | -3 to +5 |

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

### Example Demo Output

```
==================================================
  TASK: EASY
==================================================
  ✓ Ep 01 | Score: 1.000 | Reward: +7.50 | Steps: 1 | Queries: 0 | Type: standard
  ✓ Ep 02 | Score: 0.889 | Reward: +5.30 | Steps: 2 | Queries: 1 | Type: standard
  ...
  --- SUMMARY (EASY) ---
  Average Score  : 0.9200
  Average Reward : +6.100

  TASK: HARD
  △ Ep 01 | Score: 0.552 | Reward: +1.20 | Steps: 3 | Queries: 2 | Type: drug_interaction
  ✗ Ep 02 | Score: 0.310 | Reward: -2.50 | Steps: 2 | Queries: 1 | Type: ambiguous
  ...
```

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
