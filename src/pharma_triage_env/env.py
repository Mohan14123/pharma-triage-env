"""
PharmaTriageEnv — Multi-step pharmacovigilance triage environment.

OpenEnv-compliant API:
  - reset()  -> Observation
  - step(action) -> (Observation, Reward, done, info)
  - state()  -> internal state dict

Multi-step interaction:
  1. Agent receives initial (partial) observation
  2. Agent can query for hidden information (up to max_steps - 1)
  3. Agent submits final triage decision
  4. Environment grades and returns shaped reward + deterministic score

The agent's Action can contain either:
  - A `query` field (to request info) → returns updated observation
  - A full decision (severity/serious/expected/escalation) → terminates episode
"""

from pharma_triage_env.grader import TriageGrader
from pharma_triage_env.reward import RewardCalculator
from pharma_triage_env.models import Observation, Action, Reward


class PharmaTriageEnv:
    """
    Multi-step pharmacovigilance triage environment.

    Supports 3 task difficulties: easy, medium, hard.
    Each episode allows up to max_steps interactions (queries + final decision).

    Args:
        task: difficulty level ("easy", "medium", "hard")
        max_steps: maximum interaction steps per episode
        seed: base RNG seed for reproducibility. When set, episode i uses
              seed = base_seed + i, guaranteeing identical cases across runs.
              When None, cases are generated randomly (non-reproducible).
    """

    def __init__(self, task="hard", max_steps=5, seed=None):
        self.task = task
        self.max_steps = max_steps
        self.seed = seed
        self.grader = TriageGrader()
        self.reward_calc = RewardCalculator()

        # episode state
        self.current_case = None
        self.step_count = 0
        self.queries_used = 0
        self.revealed_fields = {}
        self.episode_done = False
        self.episode_history = []
        self._episode_index = 0

    def reset(self, seed=None):
        """
        Reset environment for a new episode.

        Args:
            seed: optional per-episode seed. If not provided and base_seed
                  was set at init, auto-seeds as base_seed + episode_count.

        Returns:
            Observation: initial (possibly partial) observation
        """
        from pharma_triage_env.generator import generate_case

        # Deterministic seeding
        ep_seed = seed
        if ep_seed is None and self.seed is not None:
            ep_seed = self.seed + self._episode_index
            self._episode_index += 1

        self.current_case = generate_case(self.task, seed=ep_seed)
        self.step_count = 0
        self.queries_used = 0
        self.revealed_fields = {}
        self.episode_done = False
        self.episode_history = []

        obs = self._build_observation()
        return obs

    def step(self, action):
        """
        Execute one step.

        If action contains a `query`: reveal info and return updated observation.
        If action contains a decision: grade and return final reward.

        Args:
            action: Action model or dict

        Returns:
            tuple: (Observation, Reward, done: bool, info: dict)
        """
        if self.episode_done:
            return (
                self._build_observation(),
                Reward(value=0.0, breakdown={}),
                True,
                {"error": "Episode already done. Call reset()."}
            )

        # normalize action to dict
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {}

        self.step_count += 1

        # ---- QUERY path ----
        query = action_dict.get("query")
        if query and self.step_count < self.max_steps:
            return self._handle_query(query, action_dict)

        # ---- DECISION path (final step or forced by max_steps) ----
        return self._handle_decision(action_dict)

    def state(self):
        """
        Return internal environment state (for debugging/inspection).

        Returns:
            dict: full internal state
        """
        return {
            "task": self.task,
            "step_count": self.step_count,
            "queries_used": self.queries_used,
            "episode_done": self.episode_done,
            "revealed_fields": list(self.revealed_fields.keys()),
            "max_steps": self.max_steps,
            "case_metadata": self.current_case.get("case_metadata") if self.current_case else None,
            "ground_truth": self.current_case.get("ground_truth") if self.current_case and self.episode_done else "hidden",
            "episode_history": self.episode_history,
        }

    # ================================================================
    # INTERNAL METHODS
    # ================================================================

    def _build_observation(self, query_response=None):
        """Build Observation model from current case state."""
        if not self.current_case:
            raise RuntimeError("No active case. Call reset() first.")

        obs_data = dict(self.current_case["observation"])

        # merge revealed fields
        for field, value in self.revealed_fields.items():
            obs_data[field] = value

        # add multi-step metadata
        meta = self.current_case.get("case_metadata", {})
        obs_data["step_number"] = self.step_count
        obs_data["max_steps"] = self.max_steps
        obs_data["available_queries"] = meta.get("available_queries", [])
        obs_data["query_response"] = query_response

        return Observation(**obs_data)

    def _handle_query(self, query, action_dict):
        """Process an information-gathering query."""
        hidden = self.current_case.get("hidden_info", {})

        if query in hidden:
            info = hidden[query]
            # reveal the field
            self.revealed_fields[info["field"]] = info["value"]
            response_text = info["response"]
            self.queries_used += 1

            # record in history
            self.episode_history.append({
                "step": self.step_count,
                "type": "query",
                "query": query,
                "response": response_text,
            })

            # small reward for useful querying
            obs = self._build_observation(query_response=response_text)
            reward_val = 0.2 if self.queries_used <= 3 else -0.3
            return (
                obs,
                Reward(value=reward_val, breakdown={"query_step": reward_val}),
                False,
                {"query": query, "response": response_text, "step": self.step_count}
            )
        else:
            # invalid query
            self.episode_history.append({
                "step": self.step_count,
                "type": "invalid_query",
                "query": query,
            })
            obs = self._build_observation(query_response=f"Unknown query: {query}")
            return (
                obs,
                Reward(value=-0.5, breakdown={"invalid_query": -0.5}),
                False,
                {"error": f"Unknown query: {query}", "step": self.step_count}
            )

    def _handle_decision(self, pred):
        """Process the final triage decision."""
        self.episode_done = True
        gt = self.current_case["ground_truth"]
        meta = self.current_case.get("case_metadata", {})

        # clean gt of internal fields
        gt_clean = {k: v for k, v in gt.items() if not k.startswith("_")}

        # grade (deterministic score 0-1)
        score = self.grader.grade(pred, gt_clean, task=self.task, case_metadata=meta)

        # shaped reward (-10 to +10)
        reward_val, breakdown = self.reward_calc.compute(
            pred, gt_clean,
            case_metadata=meta,
            steps_taken=self.step_count,
            queries_used=self.queries_used,
        )

        self.episode_history.append({
            "step": self.step_count,
            "type": "decision",
            "prediction": pred,
            "ground_truth": gt_clean,
            "score": score,
            "reward": reward_val,
        })

        obs = self._build_observation()
        return (
            obs,
            Reward(value=reward_val, breakdown=breakdown),
            True,
            {
                "score": score,
                "reward_breakdown": breakdown,
                "steps_taken": self.step_count,
                "queries_used": self.queries_used,
                "case_type": meta.get("case_type", "standard"),
                "ground_truth": gt_clean,
            }
        )