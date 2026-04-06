"""
Demo script for PharmaTriageEnv.

Runs a deterministic baseline agent across all 3 task difficulties,
demonstrating the multi-step interaction loop.

Usage:
    PYTHONPATH=./src python scripts/demo.py
"""

from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Action


def baseline_agent(obs, step_num=0):
    """
    Deterministic baseline agent with multi-step interaction.

    Strategy:
    - Step 0-1: query for missing critical fields
    - Final step: make decision based on all available info
    """
    # decide whether to query or decide
    available = obs.available_queries or []

    # query strategy: ask for critical missing info first
    if step_num == 0 and obs.hospitalized is None and "was_patient_hospitalized" in available:
        return Action(query="was_patient_hospitalized")

    if step_num == 1 and obs.life_threatening is None and "was_event_life_threatening" in available:
        return Action(query="was_event_life_threatening")

    if step_num <= 1 and obs.known_label_side_effects is None and "is_symptom_in_drug_label" in available:
        return Action(query="is_symptom_in_drug_label")

    # make decision
    symptoms = obs.symptoms or []
    known = obs.known_label_side_effects or []

    serious = bool(obs.hospitalized) or bool(obs.life_threatening)
    expected = any(s in known for s in symptoms)

    # check for severe symptoms
    severe_keywords = ["anaphylaxis", "cardiac arrest", "hemorrhagic", "rhabdomyolysis",
                       "Stevens-Johnson", "serotonin syndrome", "aplastic anemia",
                       "pulmonary fibrosis", "pancytopenia", "aortic dissection",
                       "GI bleeding", "renal failure", "hepatotoxicity", "skin necrosis"]

    has_severe = any(kw.lower() in s.lower() for s in symptoms for kw in severe_keywords)

    if has_severe:
        severity = "high"
        serious = True  # override
    elif serious:
        severity = "medium"
    else:
        severity = "low"

    if serious and not expected:
        escalation = "regulatory_report"
    elif serious:
        escalation = "urgent_review"
    else:
        escalation = "routine_review"

    return Action(
        severity=severity,
        serious=serious,
        expected=expected,
        escalation=escalation,
    )


def run_demo():
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*50}")

        env = PharmaTriageEnv(task=task, max_steps=5)

        scores = []
        rewards = []
        total_queries = 0
        total_steps = 0

        num_episodes = 25

        for i in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0

            for step_num in range(env.max_steps):
                action = baseline_agent(obs, step_num)

                obs, reward, done, info = env.step(action)
                reward_val = reward.value if hasattr(reward, "value") else 0.0
                episode_reward += reward_val

                if done:
                    score = info.get("score", 0.0)
                    queries = info.get("queries_used", 0)
                    steps = info.get("steps_taken", step_num + 1)
                    case_type = info.get("case_type", "standard")

                    scores.append(score)
                    rewards.append(episode_reward)
                    total_queries += queries
                    total_steps += steps

                    status = "✓" if score >= 0.7 else "△" if score >= 0.4 else "✗"
                    print(
                        f"  {status} Ep {i+1:02d} | "
                        f"Score: {score:.3f} | "
                        f"Reward: {episode_reward:+6.2f} | "
                        f"Steps: {steps} | "
                        f"Queries: {queries} | "
                        f"Type: {case_type}"
                    )
                    break

        avg_score = sum(scores) / len(scores)
        avg_reward = sum(rewards) / len(rewards)
        avg_queries = total_queries / num_episodes
        avg_steps = total_steps / num_episodes

        print(f"\n  --- SUMMARY ({task.upper()}) ---")
        print(f"  Average Score  : {avg_score:.4f}")
        print(f"  Average Reward : {avg_reward:+.3f}")
        print(f"  Avg Queries    : {avg_queries:.1f}")
        print(f"  Avg Steps      : {avg_steps:.1f}")
        print(f"  Min/Max Score  : {min(scores):.3f} / {max(scores):.3f}")

    print(f"\n{'='*50}")
    print("  Demo completed successfully.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_demo()