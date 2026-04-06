"""
FastAPI server for PharmaTriageEnv.

Endpoints:
  GET  /                — root info / health page
  GET  /health          — health check (JSON)
  GET  /tasks           — list available tasks
  POST /reset           — HTTP reset (dev only; not accessible on HF Spaces)
  POST /step            — HTTP step  (dev only; not accessible on HF Spaces)
  GET  /state           — HTTP state (dev only)
  WS   /ws              — WebSocket endpoint (primary; required for HF Spaces)

Scaling env vars:
  WORKERS              = 2      (uvicorn worker processes; 2 on HF free tier)
  MAX_CONCURRENT_ENVS  = 100    (max WebSocket sessions per worker)
  PORT                 = 7860
  HOST                 = 0.0.0.0
"""

import asyncio
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pharma_triage_env.env import PharmaTriageEnv
from pharma_triage_env.models import Action
from pharma_triage_env.tasks import TASKS

MAX_CONCURRENT_ENVS = int(os.environ.get("MAX_CONCURRENT_ENVS", 100))

app = FastAPI(
    title="PharmaTriageEnv API",
    description="Multi-step pharmacovigilance triage environment",
    version="1.0.0",
)

# HTTP session storage (dev / single-user only)
sessions: dict = {}

# Active WebSocket session counter
_ws_session_count = 0
_ws_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Root page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Human-readable landing page."""
    task_list = "".join(f"<li><code>{k}</code> — {v.get('description','')}</li>" for k, v in TASKS.items())
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8"/>
      <title>PharmaTriageEnv</title>
      <style>
        body {{ font-family: sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }}
        h1 {{ color: #1a5276; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }}
        .badge {{ background: #27ae60; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
      </style>
    </head>
    <body>
      <h1>💊 PharmaTriageEnv <span class="badge">Running</span></h1>
      <p>Multi-step Adverse Drug Event triage environment compatible with
         <a href="https://huggingface.co/openenv">OpenEnv</a>.</p>
      <h2>Connection</h2>
      <ul>
        <li><strong>WebSocket (primary):</strong> <code>ws://&lt;host&gt;/ws</code></li>
        <li><strong>HTTP (dev only):</strong> <code>/reset</code>, <code>/step</code>, <code>/state</code></li>
        <li><strong>Docs:</strong> <a href="/docs">/docs</a></li>
      </ul>
      <h2>Available Tasks</h2>
      <ul>{task_list}</ul>
      <h2>WebSocket Protocol</h2>
      <p>Send JSON messages with a <code>type</code> field:</p>
      <pre>
  {{"type": "reset", "task": "hard", "max_steps": 5}}
  {{"type": "step",  "query": "was_patient_hospitalized"}}
  {{"type": "step",  "severity": "high", "serious": true, "expected": false, "escalation": "regulatory_report"}}
  {{"type": "state"}}
      </pre>
      <p>Max concurrent WebSocket sessions: <strong>{MAX_CONCURRENT_ENVS}</strong></p>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "PharmaTriageEnv",
        "max_concurrent_envs": MAX_CONCURRENT_ENVS,
        "active_ws_sessions": _ws_session_count,
    }


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    return TASKS


# ---------------------------------------------------------------------------
# HTTP endpoints (dev / local use — NOT accessible on HF Spaces)
# ---------------------------------------------------------------------------

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


@app.post("/reset")
def reset(req: ResetRequest):
    """Reset environment for a new episode (HTTP — dev only)."""
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
    """Submit an action (HTTP — dev only)."""
    env = sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")

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
    """Get internal environment state (HTTP — dev only)."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state()


# ---------------------------------------------------------------------------
# WebSocket endpoint — primary interface for HF Spaces
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint. Each connection gets an isolated PharmaTriageEnv instance.

    Message format (client → server):
      {"type": "reset", "task": "hard", "max_steps": 5}
      {"type": "step", "query": "was_patient_hospitalized"}
      {"type": "step", "severity": "high", "serious": true, "expected": false, "escalation": "regulatory_report"}
      {"type": "state"}
      {"type": "tasks"}

    Response format (server → client):
      {"type": "reset",   "observation": {...}, "task": ..., "max_steps": ...}
      {"type": "step",    "observation": {...}, "reward": {...}, "done": bool, "info": {...}}
      {"type": "state",   "state": {...}}
      {"type": "tasks",   "tasks": {...}}
      {"type": "error",   "message": "..."}
      {"type": "limit",   "message": "Max concurrent sessions reached"}
    """
    global _ws_session_count

    async with _ws_lock:
        if _ws_session_count >= MAX_CONCURRENT_ENVS:
            await websocket.accept()
            await websocket.send_json({
                "type": "limit",
                "message": f"Max concurrent sessions ({MAX_CONCURRENT_ENVS}) reached. Try again later.",
            })
            await websocket.close(code=1013)
            return
        _ws_session_count += 1

    await websocket.accept()

    # Each WebSocket connection has its own isolated env instance
    env: Optional[PharmaTriageEnv] = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type")

            if msg_type == "reset":
                task = data.get("task", "hard")
                max_steps = int(data.get("max_steps", 5))
                if task not in TASKS:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid task '{task}'. Choose from {list(TASKS.keys())}",
                    })
                    continue
                env = PharmaTriageEnv(task=task, max_steps=max_steps)
                obs = env.reset()
                await websocket.send_json({
                    "type": "reset",
                    "observation": obs.model_dump(),
                    "task": task,
                    "max_steps": max_steps,
                })

            elif msg_type == "step":
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Send a 'reset' message before stepping.",
                    })
                    continue
                action = Action(
                    severity=data.get("severity"),
                    serious=data.get("serious"),
                    expected=data.get("expected"),
                    escalation=data.get("escalation"),
                    query=data.get("query"),
                )
                obs, reward, done, info = env.step(action)
                await websocket.send_json({
                    "type": "step",
                    "observation": obs.model_dump(),
                    "reward": reward.model_dump(),
                    "done": done,
                    "info": info,
                })

            elif msg_type == "state":
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No active environment. Send 'reset' first.",
                    })
                    continue
                await websocket.send_json({"type": "state", "state": env.state()})

            elif msg_type == "tasks":
                await websocket.send_json({"type": "tasks", "tasks": TASKS})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type '{msg_type}'. Use: reset, step, state, tasks.",
                })

    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_session_count -= 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    workers = int(os.environ.get("WORKERS", 2))
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()