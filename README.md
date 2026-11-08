# 🤖 manipulation-policy-pipeline

> **Imitation-learning policy (BC + Diffusion Policy) with a full production deployment pipeline**

An end-to-end robotics ML project: trains a manipulation policy using Behavioural Cloning and Diffusion Policy, then deploys it as a production-grade service with FastAPI, Docker, Kubernetes, and Prometheus/Grafana monitoring — demonstrating the full MLOps lifecycle from training to live observability.

---

## ✨ What this demonstrates

- **Two policy architectures** — Behavioural Cloning (BC) and Diffusion Policy, trained and compared
- **Full MLOps stack** — the same model that trained locally ships to Kubernetes
- **Observability** — Prometheus metrics + Grafana dashboard for inference latency, throughput, error rate
- **CI/CD** — GitHub Actions pipeline: lint → test → build Docker image → push → deploy

---

## 🏗️ Architecture

```
Training
  └── BC Policy / Diffusion Policy (PyTorch)
        └── Saved checkpoint

Serving
  └── FastAPI inference server
        └── Docker image
              └── Kubernetes Deployment
                    ├── HPA (auto-scaling)
                    └── Service (LoadBalancer)

Monitoring
  └── Prometheus (metrics scrape)
        └── Grafana (dashboard)
```

---

## 🚀 Quick start

**Local training:**
```bash
git clone https://github.com/Nawfal-AI/manipulation-policy-pipeline
cd manipulation-policy-pipeline
pip install -r requirements.txt
python train.py --policy bc --config configs/bc.yaml
```

**Docker:**
```bash
docker build -t manipulation-policy .
docker run -p 8000:8000 manipulation-policy
```

**Kubernetes:**
```bash
kubectl apply -f k8s/
# Grafana dashboard: kubectl port-forward svc/grafana 3000:3000
```

---

## 🔧 Stack

| Component | Technology |
|---|---|
| Policy training | PyTorch · NumPy |
| Serving | FastAPI |
| Containerisation | Docker |
| Orchestration | Kubernetes · Helm |
| Monitoring | Prometheus · Grafana |
| CI/CD | GitHub Actions |

---

## 📂 Structure

```
manipulation-policy-pipeline/
├── train.py
├── policies/
│   ├── bc_policy.py
│   └── diffusion_policy.py
├── serve/
│   └── main.py               # FastAPI app
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboard.json
├── .github/workflows/ci.yml
└── Dockerfile
```

---

## 📄 License

MIT
