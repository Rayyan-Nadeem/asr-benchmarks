#!/usr/bin/env bash
# Commit measurement results back to the repo so the Mac side can pull them.
# Cloud-agnostic — runs anywhere git is installed.
#
# Reads from cloud/.env (or env vars directly):
#   GITHUB_PAT          — PAT with `repo` scope, https://github.com/settings/tokens
#   GIT_REPO            — e.g., Rayyan-Nadeem/asr-benchmarks
#   GIT_BRANCH          — default: current branch
#   GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL — commit author identity
#
# Usage on the GPU box, after run_experiments.sh finishes:
#   bash cloud/push_results.sh "GPU session: NeMo FP16 + vocab bias + KenLM measurements"

set -euo pipefail
cd "$(dirname "$0")/.."

# Source .env if present (don't require it — env vars work too)
if [ -f "cloud/.env" ]; then
  set -a; . cloud/.env; set +a
fi

: "${GITHUB_PAT:?GITHUB_PAT not set (see cloud/.env.example)}"
: "${GIT_REPO:?GIT_REPO not set (e.g. Rayyan-Nadeem/asr-benchmarks)}"

COMMIT_MSG="${1:-GPU measurement session results}"
BRANCH="${GIT_BRANCH:-$(git branch --show-current)}"

# Configure git identity on the box (no-op if already set)
git config user.name "${GIT_AUTHOR_NAME:-asr-benchmarks-bot}"
git config user.email "${GIT_AUTHOR_EMAIL:-noreply@asr-benchmarks}"

# Update the remote to embed the PAT so we can push from this box without
# needing SSH keys configured. Doesn't persist beyond this script's life.
# We restore the original URL on exit so we don't leave the PAT in the
# repo's git config.
ORIG_URL=$(git remote get-url origin)
trap 'git remote set-url origin "${ORIG_URL}" 2>/dev/null || true' EXIT
git remote set-url origin "https://${GITHUB_PAT}@github.com/${GIT_REPO}.git"

# Stage everything new under results/ and any updated scoreboard.
git add results/runs/*.json results/runs/*.jsonl 2>/dev/null || true
git add results/SCOREBOARD.md 2>/dev/null || true

# Bail if nothing changed (script was run with no new measurements).
if git diff --cached --quiet; then
  echo "Nothing to commit — no new measurement files staged."
  exit 0
fi

# Commit + push
git commit -m "${COMMIT_MSG}

Generated on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ).
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'gpu: unknown')

New runs: $(git diff --cached --name-only | grep -c 'results/runs/.*\.json$' || true) JSON files"

git push origin "${BRANCH}"
echo "Pushed to origin/${BRANCH}"
