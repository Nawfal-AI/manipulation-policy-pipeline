"""
FastAPI inference server: the deployment-facing wrapper around a trained
policy checkpoint. This is the piece that plays the role of "the interface a
Forward Deployed Engineer builds between a customer's model and their robot's
control loop" — a well-defined API, health checks, and metrics, not just a
model in a notebook.
"""
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.responses import Response

from models import BCPolicy, DiffusionPolicy

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/bc_policy.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Manipulation Policy Inference Server")

REQUEST_COUNT = Counter("policy_requests_total", "Total prediction requests", ["status"])
LATENCY = Histogram("policy_inference_latency_seconds", "Inference latency in seconds")

_state = {"model": None, "obs_mean": None, "obs_std": None, "policy_type": None}


class ObservationRequest(BaseModel):
    observation: list[float]  # length-8 vector: [ee(2), object(2), goal(2), object-goal(2)]


class ActionResponse(BaseModel):
    action: list[float]
    policy_type: str
    latency_ms: float


@app.on_event("startup")
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        # Don't crash on startup with no checkpoint present (e.g. first container
        # build before training has run) — fail loudly and clearly on /predict
        # instead, so `docker run` for a smoke test doesn't require a checkpoint.
        print(f"WARNING: checkpoint not found at {CHECKPOINT_PATH}. /predict will 503 until it exists.")
        return
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = BCPolicy() if ckpt["policy_type"] == "bc" else DiffusionPolicy()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    _state["model"] = model
    _state["obs_mean"] = ckpt["obs_mean"].to(DEVICE)
    _state["obs_std"] = ckpt["obs_std"].to(DEVICE)
    _state["policy_type"] = ckpt["policy_type"]
    print(f"Loaded {ckpt['policy_type']} policy from {CHECKPOINT_PATH} on {DEVICE}")


@app.get("/health")
def health():
    return {"status": "ok" if _state["model"] is not None else "no_checkpoint_loaded", "device": DEVICE}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=ActionResponse)
def predict(req: ObservationRequest):
    if _state["model"] is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=f"No checkpoint loaded from {CHECKPOINT_PATH}")
    if len(req.observation) != 8:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="observation must have length 8")

    start = time.perf_counter()
    obs_t = torch.tensor(req.observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    obs_t = (obs_t - _state["obs_mean"]) / _state["obs_std"]
    with LATENCY.time():
        action = _state["model"].act(obs_t).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000

    REQUEST_COUNT.labels(status="success").inc()
    return ActionResponse(action=action.tolist(), policy_type=_state["policy_type"], latency_ms=latency_ms)
