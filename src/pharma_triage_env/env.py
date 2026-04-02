from pharma_triage_env.grader import TriageGrader
from pharma_triage_env.reward import RewardCalculator
from pharma_triage_env.models import Observation, Action, Reward


class PharmaTriageEnv:
    def __init__(self, task="hard"):
        self.task = task
        self.grader = TriageGrader()
        self.reward_calc = RewardCalculator()
        self.current_case = None

    def reset(self):
        from pharma_triage_env.generator import generate_case
        self.current_case = generate_case(self.task)
        return Observation(**self.current_case["observation"])

    def step(self, action: Action):
        gt = self.current_case["ground_truth"]

        pred = action.model_dump() if hasattr(action, "model_dump") else {}

        # ✅ FINAL SCORE (for evaluation)
        score = self.grader.grade(pred, gt, task=self.task)

        # ✅ SHAPED REWARD (for learning signal)
        reward_val = self.reward_calc.compute(pred, gt)

        return (
            Observation(**self.current_case["observation"]),
            Reward(value=reward_val),
            True,
            {"score": score}
        )