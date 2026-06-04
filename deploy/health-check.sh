#!/bin/sh
# Container health check — hits /ready on the local server.
# Used by Docker's HEALTHCHECK instruction in the Dockerfile.
#
# Returns 0 if the server is up and the engine has finished loading;
# non-zero otherwise. Docker uses this to mark the container as healthy
# in `docker ps` output and to gate dependent service startup.

set -eu

PORT="${PORT:-9000}"
URL="http://127.0.0.1:${PORT}/ready"

# Single short-timeout probe. Don't retry inside the script — let Docker's
# retry policy (Dockerfile HEALTHCHECK --retries) handle that. Multiple
# script-internal retries hide problems and inflate the HEALTHCHECK budget.
exec curl --silent --show-error --fail --max-time 5 "$URL" >/dev/null
