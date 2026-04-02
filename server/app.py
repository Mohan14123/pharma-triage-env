from fastapi import FastAPI
from pharma_triage_env.env import PharmaTriageEnv

app = FastAPI()
env = PharmaTriageEnv()

@app.get("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


# 🔥 THIS LINE WAS MISSING (critical)
if __name__ == "__main__":
    main()