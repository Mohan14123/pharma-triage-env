"""
Shaped reward calculator for PharmaTriageEnv.

Range: [-10.0, +10.0]
NOT binary — rewards partial progress, penalizes dangerous misses,
over-escalation, invalid actions, and information-gathering loops.

Reward components:
  - Base correctness (per-field)
  - Safety-critical bonuses/penalties
  - Escalation alignment
  - Information-gathering reward/penalty
  - Complexity scaling
"""


class RewardCalculator:
    """Compute shaped reward signal in [-10, +10]."""

    def compute(self, pred, gt, case_metadata=None, steps_taken=1, queries_used=0):
        """
        Args:
            pred: dict with agent predictions
            gt: dict with ground truth
            case_metadata: dict with case_type, complexity, etc.
            steps_taken: number of env steps taken so far
            queries_used: number of info-gathering queries used

        Returns:
            tuple: (reward_value, breakdown_dict)
        """
        breakdown = {}
        reward = 0.0

        meta = case_metadata or {}
        complexity = meta.get("complexity", 1.0)

        # ============================================================
        # 1. BASE CORRECTNESS (up to +5.0)
        # ============================================================

        # Severity: +1.5 correct, -0.5 wrong, +0.5 partial (off-by-one)
        severity_map = {"low": 0, "medium": 1, "high": 2}
        pred_sev = severity_map.get(pred.get("severity"), -1)
        gt_sev = severity_map.get(gt.get("severity"), -1)

        if pred_sev == gt_sev:
            breakdown["severity"] = 1.5
        elif abs(pred_sev - gt_sev) == 1 and pred_sev >= 0:
            breakdown["severity"] = 0.5    # partial credit for adjacent
        else:
            breakdown["severity"] = -0.5

        # Seriousness: +1.5 correct, -1.5 wrong (binary, high-stakes)
        if pred.get("serious") == gt.get("serious"):
            breakdown["serious"] = 1.5
        else:
            breakdown["serious"] = -1.5

        # Expectedness: +1.0 correct, -0.5 wrong
        if pred.get("expected") == gt.get("expected"):
            breakdown["expected"] = 1.0
        else:
            breakdown["expected"] = -0.5

        # Escalation: +1.0 correct, partial otherwise
        esc_order = {"routine_review": 0, "urgent_review": 1, "regulatory_report": 2}
        pred_esc = esc_order.get(pred.get("escalation"), -1)
        gt_esc = esc_order.get(gt.get("escalation"), -1)

        if pred_esc == gt_esc:
            breakdown["escalation"] = 1.0
        elif abs(pred_esc - gt_esc) == 1 and pred_esc >= 0:
            breakdown["escalation"] = 0.3    # off-by-one partial credit
        else:
            breakdown["escalation"] = -1.0

        # ============================================================
        # 2. SAFETY-CRITICAL BONUSES (up to +4.0)
        # ============================================================

        # Correctly catching serious + unexpected → big bonus
        if gt.get("serious") and not gt.get("expected"):
            if pred.get("serious") and not pred.get("expected"):
                breakdown["safety_catch"] = 3.0   # critical detection
            elif pred.get("serious"):
                breakdown["safety_catch"] = 1.0   # caught seriousness at least
            else:
                breakdown["safety_catch"] = 0.0
        else:
            breakdown["safety_catch"] = 0.0

        # Correct high-severity detection for severe symptoms
        if gt.get("severity") == "high" and pred.get("severity") == "high":
            breakdown["severity_bonus"] = 1.0
        else:
            breakdown["severity_bonus"] = 0.0

        # ============================================================
        # 3. SAFETY PENALTIES (down to -6.0)
        # ============================================================

        # Missing a serious case is DANGEROUS
        if gt.get("serious") and not pred.get("serious"):
            breakdown["missed_serious"] = -4.0

            # even worse if it was unexpected (novel ADE)
            if not gt.get("expected"):
                breakdown["missed_novel_ade"] = -2.0
            else:
                breakdown["missed_novel_ade"] = 0.0
        else:
            breakdown["missed_serious"] = 0.0
            breakdown["missed_novel_ade"] = 0.0

        # Over-escalation penalty (less severe than under-escalation)
        if pred_esc > gt_esc and pred_esc >= 0 and gt_esc >= 0:
            breakdown["over_escalation"] = -1.0 * (pred_esc - gt_esc)
        else:
            breakdown["over_escalation"] = 0.0

        # Under-escalation is worse
        if pred_esc < gt_esc and pred_esc >= 0 and gt_esc >= 0:
            breakdown["under_escalation"] = -1.5 * (gt_esc - pred_esc)
        else:
            breakdown["under_escalation"] = 0.0

        # ============================================================
        # 4. INFORMATION-GATHERING SIGNAL
        # ============================================================

        # Reward efficient querying (multi-step)
        if queries_used > 0 and queries_used <= 3:
            breakdown["query_efficiency"] = 0.5   # good info-gathering
        elif queries_used > 3:
            breakdown["query_penalty"] = -0.5 * (queries_used - 3)   # looping
            breakdown["query_efficiency"] = 0.0
        else:
            breakdown["query_efficiency"] = 0.0
            breakdown.setdefault("query_penalty", 0.0)

        # Step penalty — discourage wasting steps
        if steps_taken > 3:
            breakdown["step_penalty"] = -0.3 * (steps_taken - 3)
        else:
            breakdown["step_penalty"] = 0.0

        # ============================================================
        # 5. INVALID ACTION PENALTY
        # ============================================================
        valid_severities = {"low", "medium", "high"}
        valid_escalations = {"routine_review", "urgent_review", "regulatory_report"}

        invalid_penalty = 0.0
        if pred.get("severity") not in valid_severities:
            invalid_penalty -= 2.0
        if pred.get("escalation") not in valid_escalations:
            invalid_penalty -= 2.0
        if not isinstance(pred.get("serious"), bool):
            invalid_penalty -= 1.0
        if not isinstance(pred.get("expected"), bool):
            invalid_penalty -= 1.0
        breakdown["invalid_action"] = invalid_penalty

        # ============================================================
        # 6. COMPLEXITY SCALING
        # ============================================================
        # Harder cases give proportionally more reward for correct answers
        base_reward = sum(breakdown.values())
        complexity_bonus = 0.0
        if base_reward > 0:
            complexity_bonus = base_reward * (complexity - 1.0) * 0.3
        breakdown["complexity_bonus"] = round(complexity_bonus, 2)

        # ============================================================
        # 7. CASE-TYPE BONUSES
        # ============================================================
        case_type = meta.get("case_type", "standard")
        if case_type == "drug_interaction" and pred.get("severity") == "high":
            breakdown["interaction_awareness"] = 0.5
        elif case_type == "ambiguous":
            # more lenient for ambiguous cases — partial credit easier
            if breakdown.get("severity", 0) < 0:
                breakdown["ambiguity_leniency"] = 0.3
            else:
                breakdown["ambiguity_leniency"] = 0.0
        else:
            breakdown.setdefault("interaction_awareness", 0.0)
            breakdown.setdefault("ambiguity_leniency", 0.0)

        # ============================================================
        # FINAL: clamp to [-10, +10]
        # ============================================================
        total = sum(breakdown.values())
        total = max(-10.0, min(10.0, total))
        total = round(total, 2)

        return total, breakdown