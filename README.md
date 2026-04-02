# PharmaTriageEnv: Adverse Drug Event Triage Environment

## Overview

**PharmaTriageEnv** is a realistic OpenEnv-compatible environment that simulates pharmacovigilance workflows. The environment evaluates an AI agent’s ability to triage adverse drug event (ADE) reports and make safety-critical decisions under uncertainty.

The system models real-world challenges such as:

* incomplete clinical data
* conflicting signals
* ambiguous symptom interpretation
* high-risk escalation decisions

---

## Motivation

Pharmacovigilance teams are responsible for identifying serious and unexpected adverse drug reactions from large volumes of reports. Missing critical cases can lead to:

* regulatory non-compliance
* delayed interventions
* patient harm

This environment provides a structured benchmark to evaluate how well AI agents handle:

* safety-critical reasoning
* decision-making under uncertainty
* structured + unstructured data interpretation

---

## Core Tasks

Each episode requires the agent to output:

```json
{
  "severity": "low | medium | high",
  "serious": true | false,
  "expected": true | false,
  "escalation": "routine_review | urgent_review | regulatory_report"
}
```

### Task Breakdown

1. **Severity Classification**

   * Clinical intensity of the event

2. **Seriousness Detection**

   * Based on hospitalization or life-threatening signals

3. **Expectedness Detection**

   * Whether symptoms align with known drug side effects

4. **Escalation Decision (Primary Objective)**

   * Final triage action with highest importance

---

## Task Difficulty Levels

| Task   | Description                                              | Difficulty |
| ------ | -------------------------------------------------------- | ---------- |
| Easy   | Complete, clean data with aligned signals                | Low        |
| Medium | Partial observability and noisy inputs                   | Moderate   |
| Hard   | Adversarial cases, conflicting signals, deceptive labels | High       |

### Expected Baseline Performance

| Task   | Expected Score |
| ------ | -------------- |
| Easy   | 0.9 – 1.0      |
| Medium | 0.7 – 0.9      |
| Hard   | 0.5 – 0.75     |

---

## Environment Design

### Observation Space

Each episode provides:

```json
{
  "drug_name": "DrugA",
  "symptoms": ["rash"],
  "hospitalized": true,
  "life_threatening": false,
  "known_label_side_effects": ["fever"],
  "free_text": "Patient reports rash after medication."
}
```

Key characteristics:

* partial observability (fields may be missing)
* structured + unstructured inputs
* noisy and contradictory information

---

### Action Space

```json
{
  "severity": "...",
  "serious": ...,
  "expected": ...,
  "escalation": "..."
}
```

---

## Reward vs Score (Important)

This environment separates **learning signal** from **evaluation metric**:

### Reward (Shaped Signal)

* Dense feedback for partial correctness
* Safety-aware penalties
* Range: approximately **-2 to +4**

### Score (Final Metric)

* Deterministic grader output
* Normalized to **0.0 – 1.0**
* Used for evaluation and benchmarking

---

## Reward Design

The reward function includes:

### Positive Signals

* correct severity classification
* correct seriousness detection
* correct expectedness reasoning
* correct escalation decision

### Safety Bonuses

* correctly identifying serious + unexpected cases

### Penalties

* missing serious cases
* incorrect escalation
* over-escalation

---

## Dataset Strategy

The environment uses a **synthetic case generator** with:

* controlled randomness
* deterministic ground truth
* adversarial scenario injection

### Features

* partial missing data
* deceptive expectedness
* escalation traps
* conflicting clinical signals
* misleading free-text descriptions

This ensures:

* reproducibility
* scalability
* non-trivial evaluation

---

## Architecture

```
pharma-triage-env/
├── src/pharma_triage_env/
│   ├── env.py
│   ├── generator.py
│   ├── grader.py
│   ├── reward.py
│   ├── models.py
│
├── scripts/demo.py
├── inference.py
├── server/app.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run demo

```bash
PYTHONPATH=./src python scripts/demo.py
```

### Example Output

```
TASK: EASY → Score ~1.0
TASK: MEDIUM → Score ~0.8
TASK: HARD → Score ~0.6
```

---

## Baseline Agent

A deterministic baseline agent is included:

* handles missing data safely
* follows simplified clinical logic
* produces reproducible results

---

## OpenEnv Compliance

This environment fully implements:

* Typed Observation, Action, Reward models
* `reset()`, `step()`, `state()` API
* `openenv.yaml` manifest
* deterministic grading system

---

## Deployment

### Environment Variables

```
API_BASE_URL
MODEL_NAME
HF_TOKEN
```

### Docker

```bash
docker build -t pharma-env .
docker run -p 8000:8000 pharma-env
```

---

## Evaluation Alignment

This project is designed to maximize:

* **Real-world utility** → pharmacovigilance domain
* **Task quality** → deterministic grading + difficulty scaling
* **Environment design** → noise, ambiguity, adversarial cases
* **Code quality** → modular, reproducible, spec-compliant
* **Creativity** → safety-critical decision benchmark

---

## Limitations

* Single-step decision process (no follow-up queries)
* Simplified clinical reasoning rules
* Synthetic data (not real patient reports)

---

## Future Work

* multi-step interaction (clarification queries)
* integration with real pharmacovigilance datasets
* temporal event tracking
* improved clinical realism

---

## Conclusion

PharmaTriageEnv provides a structured, safety-aware, and non-trivial benchmark for evaluating AI agents in healthcare-inspired workflows. It balances realism, determinism, and difficulty, making it suitable for both research and evaluation.

---
