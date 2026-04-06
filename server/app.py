"""
FastAPI server for PharmaTriageEnv.

Endpoints:
  POST /reset          — reset env for a new episode
  POST /step           — submit action (query or decision)
  GET  /state          — get internal env state
  GET  /tasks          — list available tasks
  GET  /health         — health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.tasks import TASKS

app = FastAPI(
    title="PharmaTriageEnv API",
    description="Multi-step pharmacovigilance triage environment",
    version="1.0.0",
)

# Session storage (simple in-memory for single-user)
sessions = {}


class ResetRequest(BaseModel):
    task: str = "hard"
    session_id: str = "default"
    max_steps: int = 5


class StepRequest(BaseModel):
    session_id: str = "default"
    severity: Optional[str] = None
    serious: Optional[bool] = None
    expected: Optional[bool] = None
    escalation: Optional[str] = None
    query: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "environment": "PharmaTriageEnv"}


@app.get("/tasks")
def list_tasks():
    return TASKS


@app.post("/reset")
def reset(req: ResetRequest):
    """Reset environment for a new episode."""
    if req.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task: {req.task}. Choose from {list(TASKS.keys())}")

    env = PharmaTriageEnv(task=req.task, max_steps=req.max_steps)
    obs = env.reset()
    sessions[req.session_id] = env

    return {
        "observation": obs.model_dump(),
        "session_id": req.session_id,
        "task": req.task,
        "max_steps": req.max_steps,
    }


@app.post("/step")
def step(req: StepRequest):
    """Submit an action (query or decision)."""
    env = sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")

    from pharma_triage_env.models import Action

    action = Action(
        severity=req.severity,
        serious=req.serious,
        expected=req.expected,
        escalation=req.escalation,
        query=req.query,
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def get_state(session_id: str = "default"):
    """Get internal environment state."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()