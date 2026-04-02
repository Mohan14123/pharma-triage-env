class RewardCalculator:
    def compute(self, pred, gt):
        reward = 0.0

        # ========================
        # BASE SIGNAL
        # ========================

        if pred.get("severity") == gt.get("severity"):
            reward += 0.5

        if pred.get("serious") == gt.get("serious"):
            reward += 1.0

        if pred.get("expected") == gt.get("expected"):
            reward += 0.5

        if pred.get("escalation") == gt.get("escalation"):
            reward += 1.0

        # ========================
        # SAFETY BONUSES
        # ========================

        if gt.get("serious") and not gt.get("expected"):
            if pred.get("serious") and not pred.get("expected"):
                reward += 1.5  # critical detection

        # ========================
        # PENALTIES (IMPORTANT)
        # ========================

        if gt.get("serious") and not pred.get("serious"):
            reward -= 2.0  # dangerous miss

        if pred.get("escalation") == "regulatory_report" and not (
            gt.get("serious") and not gt.get("expected")
        ):
            reward -= 1.0  # over-escalation

        # ========================
        # SMOOTH CLAMP
        # ========================

        return max(-2.0, min(4.0, reward))