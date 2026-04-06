"""
Pre-deployment comprehensive test suite.
Tests: reproducibility, env API, grader, reward, output format, all tasks.
"""
import sys
import os
import json
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # for inference.py at project root

passed = 0
failed = 0
total = 0

def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")

# ============================================================
print("\n" + "="*60)
print("  PRE-DEPLOYMENT TEST SUITE")
print("="*60)

# ============================================================
print("\n🔬 1. REPRODUCIBILITY (seeded case generation)")
# ============================================================
from pharma_triage_env.generator import generate_case

for task in ["easy", "medium", "hard"]:
    c1 = generate_case(task, seed=42)
    c2 = generate_case(task, seed=42)
    c3 = generate_case(task, seed=99)
    
    test(f"[{task}] Same seed → identical drug",
         c1["observation"]["drug_name"] == c2["observation"]["drug_name"])
    test(f"[{task}] Same seed → identical symptoms",
         c1["observation"]["symptoms"] == c2["observation"]["symptoms"])
    test(f"[{task}] Same seed → identical ground truth",
         c1["ground_truth"] == c2["ground_truth"])
    test(f"[{task}] Same seed → identical free text",
         c1["observation"]["free_text"] == c2["observation"]["free_text"])
    test(f"[{task}] Same seed → identical hidden info keys",
         set(c1["hidden_info"].keys()) == set(c2["hidden_info"].keys()))
    test(f"[{task}] Different seed → different case",
         c1["observation"]["drug_name"] != c3["observation"]["drug_name"] or
         c1["observation"]["symptoms"] != c3["observation"]["symptoms"],
         "Seeds 42 and 99 produced identical cases")

# ============================================================
print("\n🔬 2. ENVIRONMENT API (reset/step/state)")
# ============================================================
from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Action, Observation, Reward

for task in ["easy", "medium", "hard"]:
    env = PharmaTriageEnv(task=task, max_steps=5, seed=100)
    
    # reset
    obs = env.reset()
    test(f"[{task}] reset() returns Observation",
         isinstance(obs, Observation))
    test(f"[{task}] Observation has drug_name",
         obs.drug_name is not None and len(obs.drug_name) > 0)
    test(f"[{task}] Observation has symptoms",
         obs.symptoms is not None and len(obs.symptoms) > 0)
    test(f"[{task}] step_number starts at 0",
         obs.step_number == 0)
    test(f"[{task}] max_steps = 5",
         obs.max_steps == 5)
    
    # query step
    if obs.available_queries:
        query = obs.available_queries[0]
        action = Action(query=query)
        obs2, reward, done, info = env.step(action)
        test(f"[{task}] Query returns Observation",
             isinstance(obs2, Observation))
        test(f"[{task}] Query returns Reward",
             isinstance(reward, Reward))
        test(f"[{task}] Query done=False",
             done == False)
        test(f"[{task}] Query response not None",
             obs2.query_response is not None,
             f"query_response was None for query '{query}'")
    
    # decision step
    action = Action(severity="medium", serious=True, expected=False, escalation="regulatory_report")
    obs3, reward, done, info = env.step(action)
    test(f"[{task}] Decision done=True",
         done == True)
    test(f"[{task}] Decision has score",
         "score" in info)
    test(f"[{task}] Score in [0, 1]",
         0.0 <= info["score"] <= 1.0,
         f"score={info.get('score')}")
    test(f"[{task}] Reward in [-10, 10]",
         -10.0 <= reward.value <= 10.0,
         f"reward={reward.value}")
    
    # state
    state = env.state()
    test(f"[{task}] state() returns dict",
         isinstance(state, dict))
    test(f"[{task}] state() has episode_done=True",
         state.get("episode_done") == True)

# ============================================================
print("\n🔬 3. ENVIRONMENT REPRODUCIBILITY (multi-episode)")
# ============================================================
env1 = PharmaTriageEnv(task="hard", max_steps=5, seed=42)
env2 = PharmaTriageEnv(task="hard", max_steps=5, seed=42)

for ep in range(5):
    o1 = env1.reset()
    o2 = env2.reset()
    test(f"Episode {ep+1} drug match",
         o1.drug_name == o2.drug_name,
         f"{o1.drug_name} != {o2.drug_name}")
    test(f"Episode {ep+1} symptoms match",
         o1.symptoms == o2.symptoms)

# ============================================================
print("\n🔬 4. GRADER DETERMINISM")
# ============================================================
from pharma_triage_env.grader import TriageGrader

grader = TriageGrader()
gt = {"severity": "high", "serious": True, "expected": False, "escalation": "regulatory_report"}

# Perfect prediction
score_perfect = grader.grade(gt, gt, task="hard")
test("Perfect prediction score > 0.9",
     score_perfect > 0.9, f"score={score_perfect}")

# Same inputs → same score
score_again = grader.grade(gt, gt, task="hard")
test("Grader deterministic (same input → same score)",
     score_perfect == score_again)

# Wrong prediction
pred_wrong = {"severity": "low", "serious": False, "expected": True, "escalation": "routine_review"}
score_wrong = grader.grade(pred_wrong, gt, task="hard")
test("Wrong prediction score < perfect",
     score_wrong < score_perfect, f"wrong={score_wrong} vs perfect={score_perfect}")

# Partial credit
pred_partial = {"severity": "medium", "serious": True, "expected": False, "escalation": "urgent_review"}
score_partial = grader.grade(pred_partial, gt, task="hard")
test("Partial prediction score between wrong and perfect",
     score_wrong < score_partial < score_perfect,
     f"wrong={score_wrong}, partial={score_partial}, perfect={score_perfect}")

# ============================================================
print("\n🔬 5. REWARD CALCULATOR")
# ============================================================
from pharma_triage_env.reward import RewardCalculator

calc = RewardCalculator()
r_perfect, bd = calc.compute(gt, gt, case_metadata={"complexity": 2.0, "case_type": "standard"})
test("Perfect reward > 0", r_perfect > 0, f"reward={r_perfect}")
test("Reward in [-10, 10]", -10.0 <= r_perfect <= 10.0)
test("Breakdown is dict", isinstance(bd, dict))
test("Breakdown has severity", "severity" in bd)
test("Breakdown has serious", "serious" in bd)

r_wrong, _ = calc.compute(pred_wrong, gt, case_metadata={"complexity": 2.0, "case_type": "standard"})
test("Wrong prediction reward < perfect", r_wrong < r_perfect)

# ============================================================
print("\n🔬 6. OUTPUT FORMAT COMPLIANCE")
# ============================================================
# Simulate what inference.py outputs
from inference import log_start, log_step, log_end

buf = io.StringIO()
with redirect_stdout(buf):
    log_start(task="easy", env="pharma-triage-env", model="test-model")
    log_step(step=1, action="query:test", reward=0.20, done=False, error=None)
    log_step(step=2, action="decide(high,true,false,regulatory_report)", reward=7.50, done=True, error=None)
    log_end(success=True, steps=2, score=0.920, rewards=[0.20, 7.50])

output = buf.getvalue()
lines = output.strip().split("\n")

test("Output has exactly 4 lines", len(lines) == 4, f"got {len(lines)} lines")
test("[START] line format",
     lines[0].startswith("[START]") and "task=" in lines[0] and "env=" in lines[0] and "model=" in lines[0])
test("[STEP] line 1 format",
     lines[1].startswith("[STEP]") and "step=" in lines[1] and "reward=" in lines[1] and "done=" in lines[1])
test("[STEP] line 2 format",
     lines[2].startswith("[STEP]") and "done=true" in lines[2])
test("[END] line format",
     lines[3].startswith("[END]") and "success=" in lines[3] and "score=" in lines[3] and "rewards=" in lines[3])
test("[END] rewards formatted to 2dp",
     "rewards=0.20,7.50" in lines[3])
test("[END] score formatted to 3dp",
     "score=0.920" in lines[3])

# Error case
buf2 = io.StringIO()
with redirect_stdout(buf2):
    log_step(step=1, action="error", reward=0.0, done=True, error="Connection timeout")
output2 = buf2.getvalue()
test("[STEP] error field shows message",
     "error=Connection timeout" in output2)

# ============================================================
print("\n🔬 7. INFERENCE CONSTANTS CHECK")
# ============================================================
import inference
test("TEMPERATURE = 0", inference.TEMPERATURE == 0)
test("MAX_STEPS = 5", inference.MAX_STEPS == 5)
test("SEED is set", hasattr(inference, 'SEED') and inference.SEED is not None)
test("NUM_EPISODES = 30", inference.NUM_EPISODES == 30)
test("SUCCESS_SCORE_THRESHOLD = 0.5", inference.SUCCESS_SCORE_THRESHOLD == 0.5)
test("API_BASE_URL has default",
     inference.API_BASE_URL == "https://router.huggingface.co/v1")
test("MODEL_NAME has default",
     inference.MODEL_NAME == "Qwen/Qwen2.5-72B-Instruct")

# ============================================================
print("\n🔬 8. TASKS CONFIGURATION")
# ============================================================
from pharma_triage_env.tasks import TASKS

test("3 tasks defined", len(TASKS) == 3)
for task_name in ["easy", "medium", "hard"]:
    test(f"Task '{task_name}' exists", task_name in TASKS)
    t = TASKS[task_name]
    test(f"Task '{task_name}' has max_steps",
         "max_steps" in t and t["max_steps"] >= 3)

# ============================================================
print("\n🔬 9. MODELS VALIDATION")
# ============================================================
# Action with query
a1 = Action(query="was_patient_hospitalized")
test("Action with query only", a1.query == "was_patient_hospitalized")

# Action with decision
a2 = Action(severity="high", serious=True, expected=False, escalation="regulatory_report")
test("Action with decision fields", a2.severity == "high" and a2.serious == True)

# Observation serialization
env = PharmaTriageEnv(task="easy", seed=42)
obs = env.reset()
d = obs.model_dump()
test("Observation serializable", isinstance(d, dict))
test("Observation has all required fields",
     all(k in d for k in ["drug_name", "symptoms", "step_number", "max_steps"]))

# Reward validation
r = Reward(value=5.5, breakdown={"test": 5.5})
test("Reward model valid", r.value == 5.5)

# ============================================================
print("\n🔬 10. SERVER IMPORT CHECK")
# ============================================================
try:
    from server.app import app
    test("FastAPI app imports", True)
    test("App has /health route",
         any(r.path == "/health" for r in app.routes))
    test("App has /reset route",
         any(r.path == "/reset" for r in app.routes))
    test("App has /step route",
         any(r.path == "/step" for r in app.routes))
    test("App has /state route",
         any(r.path == "/state" for r in app.routes))
    test("App has /tasks route",
         any(r.path == "/tasks" for r in app.routes))
except Exception as e:
    test("FastAPI app imports", False, str(e))

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
if failed == 0:
    print(f"  🎉 ALL {total} TESTS PASSED — READY TO DEPLOY!")
else:
    print(f"  ⚠️  {passed}/{total} passed, {failed} FAILED")
print("="*60 + "\n")

sys.exit(0 if failed == 0 else 1)
