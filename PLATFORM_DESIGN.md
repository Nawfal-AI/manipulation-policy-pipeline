# Platform Design — Extending to a Shared Multi-Tenant ML Platform

This document describes how `manipulation-policy-pipeline` would be extended
from a single-model service into a **shared ML platform** serving multiple
teams, models, and user communities — the kind of platform CERN's IT-CD group
operates across experiments and campus services.

The current repo already implements the single-model case end-to-end
(FastAPI + Docker + Kubernetes + Prometheus/Grafana + MLflow + Ansible + CI/CD).
This document records the design decisions needed to make that multi-tenant.

---

## What changes at the platform level

### 1. Namespace isolation per team

Each experiment or service gets its own Kubernetes namespace. Resource quotas
prevent one team's workload from starving another's.

```yaml
# k8s/namespaces/atlas-team.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: atlas-team
  labels:
    team: atlas
    env: production
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: atlas-quota
  namespace: atlas-team
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.nvidia.com/gpu: "2"      # GPU quota per team
    pods: "20"
```

Each team deploys their own model server into their namespace using the same
Helm chart (see `k8s/helm/`), with their own values overriding the defaults.

---

### 2. Shared GPU pool with priority classes

CERN's heterogeneous compute mix (CPUs, GPUs, FPGAs) means GPU resources are
shared and scarce. Priority classes let high-priority inference workloads
pre-empt lower-priority batch training jobs.

```yaml
# k8s/priority-classes.yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: inference-high         # live serving — never pre-empted
value: 1000
globalDefault: false
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-batch         # background training — pre-emptible
value: 100
globalDefault: true
```

Training jobs run as `training-batch`; the inference server runs as
`inference-high`. When the cluster is under pressure, training pods are
evicted first and resume from the latest MLflow checkpoint.

---

### 3. Per-team model registry with access control

The current MLflow setup uses a single shared registry. At platform scale,
each team needs their own experiment namespace in MLflow, with RBAC so
Team A cannot promote Team B's models to Production.

```
MLflow
├── experiments/
│   ├── atlas/manipulation-policy      ← Team A's runs
│   ├── cms/trigger-classifier         ← Team B's runs
│   └── campus/anomaly-detector        ← Campus services
└── registry/
    ├── atlas-models/                  ← Team A can promote here
    └── cms-models/                    ← Team B can promote here
```

Implementation: MLflow's permission plugin (or a thin proxy) checks the
requesting service account's namespace label before allowing a registry write.

---

### 4. Shared observability stack

Each team's model server scrapes to the same Prometheus instance using
namespace labels as the discriminator. Grafana dashboards filter by
`namespace` so teams see only their own traffic by default, but platform
operators can see everything.

```yaml
# monitoring/prometheus.yml (platform version)
scrape_configs:
  - job_name: "ml-services"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
    # Scrape any pod with annotation prometheus.io/scrape: "true"
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: "true"
```

This replaces the static `targets: ["policy-server:8000"]` in the current
`monitoring/prometheus.yml` with Kubernetes service discovery — no config
change needed when a new team deploys their model server.

---

### 5. Drift monitoring as a platform service

Currently `monitoring/drift/drift_monitor.py` runs as a one-off script.
At platform scale it runs as a CronJob, one per model, writing results to
a shared database and alerting via Alertmanager if drift exceeds threshold.

```yaml
# k8s/drift-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: drift-monitor-policy-server
  namespace: atlas-team
spec:
  schedule: "0 * * * *"           # hourly
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: drift-monitor
              image: ghcr.io/nawfal-ai/manipulation-policy-pipeline/policy-server:latest
              command:
                - python
                - monitoring/drift/drift_monitor.py
                - --reference
                - /data/reference.npz
                - --production
                - /data/live.npz
                - --output
                - /results/drift_report.html
```

An Alertmanager rule fires if `drift_share_drifted_columns > 0.3` for more
than two consecutive hours, paging the team on-call.

---

### 6. Deployment flow for a new team

A new experiment team onboards in four steps:

```
1. kubectl create namespace <team>
   kubectl apply -f k8s/namespaces/<team>.yaml   # quota + RBAC

2. helm install <team>-policy k8s/helm/manipulation-policy \
     --namespace <team> \
     --set image.tag=<their-image> \
     --set resources.gpu=1

3. MLflow: create experiment <team>/my-model
   CI/CD: add secrets for <team>'s registry

4. Grafana: import the shared dashboard template,
   set namespace filter to <team>
```

A new team is fully isolated, observable, and CI/CD-wired in under an hour.

---

## What is not in this document

- Authentication / SSO (CERN uses CERN SSO via OIDC — this would sit in
  front of the FastAPI server as an Istio AuthorizationPolicy or an NGINX
  ingress with oauth2-proxy)
- FPGA/accelerator scheduling (requires the Device Plugin for the specific
  FPGA vendor; the priority-class design above still applies)
- Long-term metric storage (Thanos or Cortex for multi-cluster Prometheus
  federation — out of scope for a single-cluster demo)

---

## Why this matters for CERN specifically

CERN's ML platform serves ATLAS, CMS, LHCb, and campus services simultaneously
from shared infrastructure. The design decisions above — namespace isolation,
GPU quotas, service-discovery-based scraping, and CronJob-based drift
monitoring — are the standard Kubernetes patterns for that workload. This
document records them explicitly so the design can be reviewed and critiqued,
rather than assumed.
