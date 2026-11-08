# Ansible Deployment

Automates deploying the full stack (policy server, Prometheus, Grafana, MLflow)
to any Ubuntu 22.04 machine — a fresh cloud VM or a local Docker test container.

## What it installs on the target machine
- Docker Engine + Docker Compose plugin
- All project services via `docker compose up`

## Requirements (on your MacBook — control machine only)
```bash
pip install ansible
ansible-galaxy collection install community.docker
```

## Option A — Deploy to a real server
1. Edit `inventory.ini` with your server's IP and SSH key path
2. Run:
```bash
ansible-playbook ansible/playbook.yml -i ansible/inventory.ini
```

## Option B — Test locally with Docker (no cloud needed)
This spins up a Docker container that acts as the "remote server":

```bash
# 1. Start a test Ubuntu SSH container
docker run -d --name ansible-test \
  -p 2222:22 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  rastasheep/ubuntu-sshd:22.04

# 2. Run the playbook against it
ansible-playbook ansible/playbook.yml -i ansible/inventory_local.ini

# 3. Verify services are up
curl http://localhost:8000/health

# 4. Clean up
docker stop ansible-test && docker rm ansible-test
```

## Services after deployment

| Service | URL |
|---|---|
| Policy API | `http://<server>:8000/docs` |
| Prometheus | `http://<server>:9090` |
| Grafana | `http://<server>:3000` (admin/admin) |
| MLflow UI | `http://<server>:5000` |
