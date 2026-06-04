# cloud/ — provisioning the GPU box

The benchmarks and the live demo both run on a single AWS GovCloud
g4dn.xlarge (Tesla T4, 16 GB VRAM). Currently deployed at
`depodash-lab.acmeplexus.com` behind Caddy TLS + HTTP basic auth.

Two terraform modules live here for two different patterns:

| Module | Pattern | When to use |
|---|---|---|
| `terraform/aws-uswest/` | Persistent demo box | Currently running. The customer-facing live demo. |
| `terraform/aws/` | Disposable per-session GPU | Historic — for the M2 measurement cycle. ~$3–5 per session, tear down when done. |

Cloud-agnostic by structure: the bash scripts (`bootstrap.sh`,
`run_experiments.sh`, `push_results.sh`) assume only "Ubuntu 22.04 + CUDA
12+ + NVIDIA driver." That's true on AWS DLAMI, Azure DSVM, GCP DLVM, and
Vast.ai's Ubuntu images. Adding Azure / GCP later means a new
`terraform/azure/` directory with matching `public_ip` / `ssh_command`
outputs.

---

## Spinning up a persistent demo box (the current setup)

```bash
cd cloud/terraform/aws-uswest
export TF_VAR_allowed_ssh_cidr="$(curl -fsS https://checkip.amazonaws.com)/32"
export TF_VAR_cloudflare_api_token="<paste from your password manager>"
terraform init
terraform plan -out tfplan
terraform apply tfplan

# Outputs you'll need:
terraform output public_ip
terraform output demo_url
terraform output -raw admin_password    # basic-auth password
```

Wait ~5 min for cloud-init to install the NVIDIA driver + reboot, then SSH in:

```bash
$(terraform output -raw ssh_command)
```

### Deploy the app stack on the box

The bootstrap installs the system layer (driver, Docker,
`nvidia-container-toolkit`, Caddy + Let's Encrypt + basic-auth) but
doesn't pull this repo. From your laptop:

```bash
# Run from your asr-benchmarks/ checkout.
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='batch' \
  --exclude='models' --exclude='results/archive' \
  -e "ssh -i ~/.ssh/id_ed25519" \
  ./ ubuntu@$(terraform output -raw public_ip):/opt/model-playground/
```

On the box:

```bash
cd /opt/model-playground
pip3 install --user -r server/requirements.txt flask flask-cors sherpa-onnx
# Pull the only model the orchestrator needs at boot time
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25', \
                    local_dir='models/nemotron-160ms')"
nohup python3 scripts/demo/control.py > /tmp/control.log 2>&1 < /dev/null &
sudo systemctl start caddy
```

Production deploys run via a systemd unit (`/etc/systemd/system/orchestrator.service`)
so the orchestrator survives reboot. See that unit on the live box for the
canonical service definition.

### Teardown

```bash
terraform destroy -auto-approve
```

Deletes the EIP and Cloudflare DNS record — demo URL stops resolving, bill stops.

---

## Disposable per-session GPU (historical pattern)

Used during the M2 measurement cycle when we wanted to run Phase-4/5 sweeps
on a fresh instance and not pay for idle. Total session cost ~$3–5, total
wall-clock ~3 hours including bootstrap.

```bash
cd cloud/terraform/aws
terraform apply \
    -var="aws_profile=${AWS_PROFILE}" \
    -var="aws_region=${AWS_REGION:-us-gov-east-1}" \
    -var="ssh_public_key_path=${SSH_PUBLIC_KEY_PATH:-~/.ssh/id_ed25519.pub}" \
    -var="allowed_ssh_cidr=${ALLOWED_SSH_CIDR:-0.0.0.0/0}"

PUBLIC_IP=$(terraform output -raw public_ip)
scp -i ~/.ssh/asr-benchmarks cloud/bootstrap.sh ubuntu@${PUBLIC_IP}:~/
ssh -i ~/.ssh/asr-benchmarks ubuntu@${PUBLIC_IP} \
    "GITHUB_PAT=$(grep GITHUB_PAT cloud/.env | cut -d= -f2-) \
     GIT_REPO=Rayyan-Nadeem/asr-benchmarks GIT_BRANCH=main \
     bash bootstrap.sh"
ssh -i ~/.ssh/asr-benchmarks ubuntu@${PUBLIC_IP} \
    'cd asr-benchmarks && source .venv/bin/activate && bash cloud/run_experiments.sh'
ssh -i ~/.ssh/asr-benchmarks ubuntu@${PUBLIC_IP} \
    "cd asr-benchmarks && GITHUB_PAT=... bash cloud/push_results.sh 'GPU session'"

terraform destroy -auto-approve
```

The structure: `bootstrap.sh` installs NeMo + downloads weights; the suite
runs FP16 / vocab-bias / KenLM / beam-4 / Sortformer / Multitalker /
pyannote measurement blocks; results push back via PAT-in-remote-URL.

---

## Security notes (both modules)

- Default SSH ingress is `var.allowed_ssh_cidr`. The default `0.0.0.0/0` is
  open-internet — set to your `<your-ip>/32` for habits' sake.
- IMDSv2 is required on both modules — prevents SSRF on EC2 metadata.
- The instance gets no IAM role — even if compromised it can't act in AWS.
- Root volume is encrypted (`encrypted = true`).
- For the disposable module, the GitHub PAT is passed via env var and never
  written to disk on the box.
- For the persistent module, basic-auth password is generated at
  `terraform apply` time and stored in tfstate — keep that file secret.

## Quotas

g5.xlarge / g4dn.xlarge needs a "Running On-Demand G and VT instances"
service quota in your AWS account (4 vCPU minimum for one instance). New
AWS accounts have this at 0 and will hit `VcpuLimitExceeded`. Request via:

```bash
aws service-quotas request-service-quota-increase \
    --service-code ec2 --quota-code L-DB2E81BA --desired-value 4
```

The Synergy account has this provisioned already.
