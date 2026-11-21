---

## Helm chart — deploy to any namespace in one command

The `k8s/helm/manipulation-policy/` chart packages the full serving stack
(Deployment, Service, HPA, Prometheus scrape annotations) so any team can
deploy their own isolated instance:

```bash
# Install into a team namespace (creates isolated deployment)
helm install my-policy k8s/helm/manipulation-policy \
  --namespace my-team \
  --create-namespace \
  --set image.tag=abc123 \
  --set replicaCount=3

# Upgrade (rolling update, zero downtime)
helm upgrade my-policy k8s/helm/manipulation-policy \
  --namespace my-team \
  --set image.tag=def456

# Uninstall (removes all resources for that team)
helm uninstall my-policy --namespace my-team
```

Values you can override per team: `image.tag`, `replicaCount`, `resources.*`,
`autoscaling.*`, `checkpoint.path`, `priorityClassName`.
See `k8s/helm/manipulation-policy/values.yaml` for all options.

---

## Multi-tenant platform design

[PLATFORM_DESIGN.md](PLATFORM_DESIGN.md) documents how this single-model
service extends to a **shared ML platform** serving multiple teams from one
cluster: namespace isolation, GPU quota per team, service-discovery-based
Prometheus scraping, shared MLflow registry with RBAC, and drift monitoring
as a CronJob. Relevant to any context where the same infrastructure needs to
serve several independent ML workloads simultaneously.
