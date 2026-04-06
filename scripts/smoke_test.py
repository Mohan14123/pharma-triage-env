"""
Smoke test for PharmaTriageEnv.

Quick validation that the environment works correctly:
  1. All 3 tasks can be instantiated and reset
  2. step() works for both queries and decisions
  3. Scores are in [0, 1], rewards in [-10, +10]
  4. Multi-step loop works
  5. state() returns valid dict
  6. Models (Observation, Action, Reward) serialize properly

Usage:
    PYTHONPATH=./src python scripts/smoke_test.py
"""

import sys
import traceback
from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Observation, Action, Reward
from pharma_triage_env.tasks import TASKS


def test_basic_reset_step():
    """Test basic reset + step for all tasks."""
    print("[TEST] Basic reset + step...")
    for task in ["easy", "medium", "hard"]:
        env = PharmaTriageEnv(task=task)
        obs = env.reset()

        assert isinstance(obs, Observation), f"reset() did not return Observation for {task}"
        assert obs.drug_name, f"Empty drug_name for {task}"
        assert len(obs.symptoms) > 0, f"Empty symptoms for {task}"

        action = Action(
            severity="medium",
            serious=True,
            expected=False,
            escalation="urgent_review",
        )
        obs2, reward, done, info = env.step(action)

        assert isinstance(obs2, Observation), f"step() obs not Observation for {task}"
        assert isinstance(reward, Reward), f"step() reward not Reward for {task}"
        assert done is True, f"step() should be done after decision for {task}"
        assert "score" in info, f"info missing 'score' for {task}"
        assert 0.0 <= info["score"] <= 1.0, f"score out of range for {task}: {info['score']}"
        assert -10.0 <= reward.value <= 10.0, f"reward out of range for {task}: {reward.value}"

    print("  ✓ PASSED\n")


def test_multi_step_query():
    """Test multi-step query interaction."""
    print("[TEST] Multi-step query...")
    env = PharmaTriageEnv(task="hard", max_steps=5)
    obs = env.reset()

    # query step
    available = obs.available_queries or []
    if available:
        query = available[0]
        action = Action(query=query)
        obs2, reward, done, info = env.step(action)

        assert done is False, "Query should not end episode"
        assert isinstance(obs2, Observation), "Query should return Observation"
        assert obs2.query_response is not None, "Query response should not be None"
        assert obs2.step_number == 1, f"Step number should be 1, got {obs2.step_number}"

    # final decision
    action = Action(
        severity="high",
        serious=True,
        expected=False,
        escalation="regulatory_report",
    )
    obs3, reward, done, info = env.step(action)
    assert done is True, "Decision should end episode"
    assert "score" in info
    assert "queries_used" in info

    print("  ✓ PASSED\n")


def test_state():
    """Test state() method."""
    print("[TEST] state()...")
    env = PharmaTriageEnv(task="medium")
    obs = env.reset()
    state = env.state()

    assert isinstance(state, dict), "state() should return dict"
    assert state["task"] == "medium", f"Wrong task in state: {state['task']}"
    assert state["step_count"] == 0
    assert state["episode_done"] is False
    assert state["ground_truth"] == "hidden", "Ground truth should be hidden before done"

    # after decision
    action = Action(severity="low", serious=False, expected=True, escalation="routine_review")
    env.step(action)
    state2 = env.state()
    assert state2["episode_done"] is True
    assert state2["ground_truth"] != "hidden", "Ground truth should be revealed after done"

    print("  ✓ PASSED\n")


def test_model_serialization():
    """Test that Pydantic models serialize properly."""
    print("[TEST] Model serialization...")
    obs = Observation(
        drug_name="TestDrug",
        symptoms=["rash"],
        hospitalized=True,
    )
    d = obs.model_dump()
    assert d["drug_name"] == "TestDrug"
    assert d["symptoms"] == ["rash"]
    assert d["hospitalized"] is True
    assert d["life_threatening"] is None   # optional field

    action = Action(severity="high", serious=True, expected=False, escalation="regulatory_report")
    ad = action.model_dump()
    assert ad["severity"] == "high"

    reward = Reward(value=5.5, breakdown={"test": 5.5})
    rd = reward.model_dump()
    assert rd["value"] == 5.5

    print("  ✓ PASSED\n")


def test_invalid_query():
    """Test that invalid queries are handled gracefully."""
    print("[TEST] Invalid query handling...")
    env = PharmaTriageEnv(task="easy")
    env.reset()

    action = Action(query="nonexistent_query")
    obs, reward, done, info = env.step(action)

    assert done is False, "Invalid query should not end episode"
    assert reward.value < 0, "Invalid query should have negative reward"
    assert "error" in info

    print("  ✓ PASSED\n")


def test_episode_already_done():
    """Test stepping after episode is done."""
    print("[TEST] Episode already done...")
    env = PharmaTriageEnv(task="easy")
    env.reset()

    action = Action(severity="low", serious=False, expected=True, escalation="routine_review")
    env.step(action)

    # step again
    _, reward, done, info = env.step(action)
    assert done is True
    assert "error" in info

    print("  ✓ PASSED\n")


def test_reward_range():
    """Test reward range across many episodes."""
    print("[TEST] Reward range [-10, +10]...")
    import random
    random.seed(42)

    for task in ["easy", "medium", "hard"]:
        env = PharmaTriageEnv(task=task)
        for _ in range(50):
            env.reset()
            action = Action(
                severity=random.choice(["low", "medium", "high"]),
                serious=random.choice([True, False]),
                expected=random.choice([True, False]),
                escalation=random.choice(["routine_review", "urgent_review", "regulatory_report"]),
            )
            _, reward, _, _ = env.step(action)
            assert -10.0 <= reward.value <= 10.0, f"Reward out of range: {reward.value}"

    print("  ✓ PASSED\n")


def test_tasks_definition():
    """Test TASKS dictionary."""
    print("[TEST] Tasks definition...")
    assert "easy" in TASKS
    assert "medium" in TASKS
    assert "hard" in TASKS

    for name, task in TASKS.items():
        assert "description" in task
        assert "difficulty" in task
        assert "max_steps" in task
        assert "expected_score_range" in task

    print("  ✓ PASSED\n")


def main():
    print("=" * 50)
    print("  PharmaTriageEnv Smoke Tests")
    print("=" * 50)
    print()

    tests = [
        test_basic_reset_step,
        test_multi_step_query,
        test_state,
        test_model_serialization,
        test_invalid_query,
        test_episode_already_done,
        test_reward_range,
        test_tasks_definition,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {test.__name__}")
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  All smoke tests passed! ✓\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
