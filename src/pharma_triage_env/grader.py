class TriageGrader:
    def grade(self, pred, gt, task="hard"):
        score = 0.0
        max_score = 6.0

        # Base scoring
        if pred.get("severity") == gt.get("severity"):
            score += 1

        if pred.get("serious") == gt.get("serious"):
            score += 2

        if pred.get("expected") == gt.get("expected"):
            score += 1

        if pred.get("escalation") == gt.get("escalation"):
            score += 2

        # Safety penalties
        if gt.get("serious") and not pred.get("serious"):
            score -= 4

        if gt.get("serious") and not gt.get("expected"):
            if not (pred.get("serious") and not pred.get("expected")):
                score -= 3

        if pred.get("escalation") == "regulatory_report" and not (
            gt.get("serious") and not gt.get("expected")
        ):
            score -= 2

        # Task-specific scoring
        if task == "easy":
            score = (
                (1 if pred.get("severity") == gt.get("severity") else 0)
                + (2 if pred.get("serious") == gt.get("serious") else 0)
            )
            max_score = 3.0

        elif task == "medium":
            score = (
                (1 if pred.get("severity") == gt.get("severity") else 0)
                + (2 if pred.get("serious") == gt.get("serious") else 0)
                + (1 if pred.get("expected") == gt.get("expected") else 0)
            )
            max_score = 4.0

        # Normalize
        normalized = score / max_score
        normalized = max(0.0, min(1.0, normalized))

        return normalized