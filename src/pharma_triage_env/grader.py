"""
Deterministic grader for PharmaTriageEnv.

Produces a normalized score in [0.0, 1.0] for evaluation.
Separate from shaped reward — this is the FINAL evaluation metric.

Scoring dimensions:
  - Severity classification
  - Seriousness detection (weighted heavily)
  - Expectedness reasoning
  - Escalation decision (weighted heavily)
  - Safety-critical compound checks
  - Task-adjusted weighting
"""


class TriageGrader:
    """Deterministic grader returning score ∈ [0.0, 1.0]."""

    # Per-task weight profiles
    WEIGHTS = {
        "easy": {
            "severity": 1.5,
            "serious": 2.5,
            "expected": 1.0,
            "escalation": 2.0,
            "safety_compound": 2.0,
            "max": 9.0,
        },
        "medium": {
            "severity": 2.0,
            "serious": 3.0,
            "expected": 1.5,
            "escalation": 2.5,
            "safety_compound": 3.0,
            "max": 12.0,
        },
        "hard": {
            "severity": 2.0,
            "serious": 3.5,
            "expected": 2.0,
            "escalation": 3.0,
            "safety_compound": 4.0,
            "max": 14.5,
        },
    }

    def grade(self, pred, gt, task="hard", case_metadata=None):
        """
        Deterministic grading.

        Args:
            pred: dict with agent predictions
            gt: dict with ground truth
            task: "easy" | "medium" | "hard"
            case_metadata: optional metadata for advanced grading

        Returns:
            float: normalized score in [0.0, 1.0]
        """
        w = self.WEIGHTS.get(task, self.WEIGHTS["hard"])
        score = 0.0

        # ========================================
        # 1. SEVERITY (exact + partial)
        # ========================================
        severity_map = {"low": 0, "medium": 1, "high": 2}
        pred_sev = severity_map.get(pred.get("severity"), -1)
        gt_sev = severity_map.get(gt.get("severity"), -1)

        if pred_sev == gt_sev:
            score += w["severity"]
        elif abs(pred_sev - gt_sev) == 1 and pred_sev >= 0:
            score += w["severity"] * 0.4    # partial credit

        # ========================================
        # 2. SERIOUSNESS (exact match, high stake)
        # ========================================
        if pred.get("serious") == gt.get("serious"):
            score += w["serious"]

        # ========================================
        # 3. EXPECTEDNESS
        # ========================================
        if pred.get("expected") == gt.get("expected"):
            score += w["expected"]

        # ========================================
        # 4. ESCALATION (exact + partial)
        # ========================================
        esc_order = {"routine_review": 0, "urgent_review": 1, "regulatory_report": 2}
        pred_esc = esc_order.get(pred.get("escalation"), -1)
        gt_esc = esc_order.get(gt.get("escalation"), -1)

        if pred_esc == gt_esc:
            score += w["escalation"]
        elif abs(pred_esc - gt_esc) == 1 and pred_esc >= 0:
            score += w["escalation"] * 0.3     # partial credit

        # ========================================
        # 5. SAFETY COMPOUND CHECK
        # ========================================
        # When the ground truth is serious + unexpected → regulatory_report
        # the agent MUST catch all three components.
        if gt.get("serious") and not gt.get("expected"):
            compound_score = 0.0
            checks = 0.0

            if pred.get("serious"):
                compound_score += 1.0
            checks += 1.0

            if not pred.get("expected"):
                compound_score += 1.0
            checks += 1.0

            if pred.get("escalation") == "regulatory_report":
                compound_score += 1.0
            checks += 1.0

            ratio = compound_score / checks
            score += w["safety_compound"] * ratio
        else:
            # Non-critical case: full safety credit for correctly NOT over-escalating.
            # A perfectly handled routine case deserves a perfect score.
            if pred.get("escalation") != "regulatory_report":
                score += w["safety_compound"]   # full credit — correct clinical judgment
            # Zero bonus if agent falsely escalates a non-critical case (penalty applied below)

        # ========================================
        # 6. PENALTIES
        # ========================================

        # Missing serious case: heavy deduction
        if gt.get("serious") and not pred.get("serious"):
            score -= w["serious"] * 0.5

        # False escalation to regulatory for non-serious cases
        if not gt.get("serious") and pred.get("escalation") == "regulatory_report":
            score -= w["escalation"] * 0.3

        # Under-escalation for serious cases
        if gt.get("serious") and pred.get("escalation") == "routine_review":
            score -= w["escalation"] * 0.4

        # ========================================
        # 7. NORMALIZE to (0.0, 1.0) exclusive
        # ========================================
        normalized = score / w["max"]
        normalized = max(0.0001, min(0.9999, normalized))
        normalized = round(normalized, 4)

        return normalized