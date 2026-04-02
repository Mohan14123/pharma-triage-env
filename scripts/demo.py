from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Action


def dummy_agent(obs):
    """
    Robust baseline agent (handles missing values)
    """

    symptoms = obs.symptoms or []
    known = obs.known_label_side_effects or []

    serious = bool(obs.hospitalized) or bool(obs.life_threatening)

    expected = any(s in known for s in symptoms)

    return Action(
        severity="high" if serious else ("medium" if symptoms else "low"),
        serious=serious,
        expected=expected,
        escalation=(
            "regulatory_report"
            if serious and not expected
            else "urgent_review"
            if serious
            else "routine_review"
        )
    )

def run_demo():
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        print(f"\n========== TASK: {task.upper()} ==========")

        env = PharmaTriageEnv(task=task)

        scores = []
        rewards = []

        for i in range(20):
            obs = env.reset()

            action = dummy_agent(obs)

            _, reward, done, info = env.step(action)

            score = info.get("score", 0.0)
            reward_val = getattr(reward, "value", 0.0)

            scores.append(score)
            rewards.append(reward_val)

            print(
                f"Run {i+1:02d} | Reward: {reward_val:+.2f} | Score: {score:.2f}"
            )

            if not done:
                print("Warning: Episode not terminated properly")

        avg_score = sum(scores) / len(scores)
        avg_reward = sum(rewards) / len(rewards)

        print("\n--- SUMMARY ---")
        print(f"Average Score : {avg_score:.3f}")
        print(f"Average Reward: {avg_reward:.3f}")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()