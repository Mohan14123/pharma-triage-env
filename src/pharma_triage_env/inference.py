from env import PharmaTriageEnv

env = PharmaTriageEnv()

def safe_bool(x):
    return False if x is None else x

def simple_agent(obs):
    symptom = obs["symptoms"][0]
    hospitalized = safe_bool(obs.get("hospitalized"))
    life_threatening = safe_bool(obs.get("life_threatening"))
    known = obs["known_label_side_effects"]

    serious = hospitalized or life_threatening
    expected = symptom in known

    if life_threatening:
        escalation = "regulatory_report"
        severity = "high"
    elif hospitalized:
        escalation = "urgent_review"
        severity = "medium"
    else:
        escalation = "routine_review"
        severity = "low"

    return {
        "severity": severity,
        "serious": serious,
        "expected": expected,
        "escalation": escalation
    }


# run multiple episodes
total_reward = 0
num_runs = 20

for i in range(num_runs):
    obs = env.reset()
    action = simple_agent(obs)
    _, reward, _, info = env.step(action)

    total_reward += reward
    print(f"Run {i+1}: Reward={reward}, Score={info}")

print("\nAverage reward:", total_reward / num_runs)