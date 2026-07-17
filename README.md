# Manipulation Policy: Training-to-Deployment Pipeline

A small, honest demonstration of the full loop a Forward Deployed Engineer
touches: train an imitation-learning policy on a manipulation-shaped task,
then wrap, containerize, monitor, and deploy it — not just a model in a
notebook.

## What this is (and isn't)

This is a **simulation-to-deployment-pattern analog**, built to learn and
demonstrate the technical pipeline hands-on. It is **not** equivalent to real
robotics deployment experience — no physical hardware, no real customer, no
production pressure. Read it as evidence of "I can build and reason about
this kind of pipeline," not "I have done this in production." See the
Honest Limitations section at the bottom before drawing conclusions from it.

Also deliberately **not MuJoCo/robosuite**: those need offscreen-rendering
dependencies that are fragile in headless Colab/Kaggle notebooks. The
environment here (`envs/push_env.py`) is a self-contained 2D pick-and-carry
task with zero heavy dependencies, so the whole pipeline runs anywhere,
including plain CPU. Swapping in a MuJoCo-based task later is a plausible
next step (see `DEPLOYMENT_RUNBOOK.md` for what that swap would need to
change).

## Pipeline stages

1. **Task & scripted expert** (`envs/push_env.py`) — a pick-and-carry
   environment: move an end-effector to an object, then carry it to a goal.
   Verified 100% scripted-expert success rate over 300 episodes
   (`test_env.py`).
2. **Demonstration generation** (`generate_demos.py`) — rolls out the expert
   with action noise for diversity, keeps only successful episodes (standard
   BC data-collection practice).
3. **Policy training** (`train_policy.py`, `models.py`) — two policies:
   - **BC**: direct regression from observation to action.
   - **Diffusion Policy**: a simplified DDPM-style conditional denoiser
     (Chi et al., 2023 pattern), which in principle handles multimodal expert
     behavior better than BC's mean-collapse failure mode — this task's
     expert is unimodal, so don't expect a dramatic difference here; the
     value is in having actually implemented the technique, not in beating
     BC on this particular task.
4. **Evaluation** (`evaluate_policy.py`) — success rate, inference latency
   (p50/p95), and robustness under observation noise. Mirrors the JD's own
   KPI list (task success rate, robustness, latency).
5. **Serving** (`serve.py`) — a FastAPI inference server: `POST /predict`,
   `/health`, `/metrics` (Prometheus format).
6. **Containerization** (`Dockerfile`) — multi-stage build, weights mounted
   as a volume (not baked into the image, so retraining doesn't require a
   rebuild).
7. **Local orchestration** (`docker-compose.yml`, `monitoring/`) — serving
   container + Prometheus + Grafana.
8. **Kubernetes** (`k8s/`) — Deployment + Service manifests, runnable on a
   local `minikube`/`k3d` cluster.
9. **CI/CD** (`.github/workflows/ci.yml`) — test on push, build and push the
   image to GHCR.
10. **Task adaptation** (`DEPLOYMENT_RUNBOOK.md`) — the obstacle-avoidance
    variant, actually implemented and tested, documenting what changed in the
    pipeline and what didn't.

## How to run

```bash
pip install -r requirements.txt

# 1. Generate demonstrations
python generate_demos.py --num_episodes 500 --output data/demos.npz

# 2. Train both policies
python train_policy.py --policy bc --data data/demos.npz
python train_policy.py --policy diffusion --data data/demos.npz

# 3. Evaluate
python evaluate_policy.py --checkpoint checkpoints/bc_policy.pt --output results/bc_eval.json
python evaluate_policy.py --checkpoint checkpoints/diffusion_policy.pt --output results/diffusion_eval.json

# 4. Serve locally
CHECKPOINT_PATH=checkpoints/bc_policy.pt uvicorn serve:app --reload
# then: curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
#   -d '{"observation": [0,0, 0.3,0.3, -0.3,-0.3, 0.6,0.6]}'

# 5. Full stack with monitoring
docker compose up --build
# policy-server: localhost:8000  |  Prometheus: localhost:9090  |  Grafana: localhost:3000

# 6. Kubernetes (local cluster)
minikube start
eval $(minikube docker-env) && docker build -t policy-server:latest .
kubectl apply -f k8s/
```

## Tests

`test_env.py` includes a regression test for a real bug caught during
development — see Honest Limitations below.

```bash
pytest test_env.py -v
```

## Honest limitations

- **Environment fidelity:** this is a 2D toy task, not a physics-accurate
  manipulation simulator. It exercises the *pipeline*, not manipulation-grade
  perception or control.
- **A real bug shaped this project's development**, worth stating plainly:
  the first version of the scripted expert (a directional pushing model)
  had a 0% success rate because the "approach offset" was smaller than the
  contact radius, so the end-effector never actually entered contact range.
  Caught by testing before shipping, fixed by switching to a simpler
  rigid-grasp dynamics model. Left as `test_scripted_expert_solves_task` in
  the test suite specifically as a regression guard.
- **No real hardware, no real customer.** See the top of this README.
- **Diffusion policy is simplified** (small MLP denoiser, no action-chunking,
  no receding horizon) — appropriate for this task's low-dimensional
  observation, not representative of image-conditioned diffusion policies at
  full scale.
