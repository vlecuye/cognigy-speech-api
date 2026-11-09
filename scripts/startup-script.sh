#!/bin/bash
# =====================================================================
# Cognigy Gemini Speech Adapter - Production Startup & Update Script
# =====================================================================
# Features:
# - Idempotent network creation (eliminates "network already exists" error)
# - Force-releases stale endpoints before launching new containers
# - Guaranteed fresh pull (bypasses Docker's local :latest cache)
# - Zero-downtime Caddy SSL reload when only the app container updates
# - Configured for ultra-low latency Google Speech Telephony STT & Streaming TTS
# =====================================================================

set -u

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
REGISTRY_REGION="europe-west1"
# Target image: using :latest, or pin to an exact @sha256: digest
APP_IMAGE="${APP_IMAGE:-europe-west1-docker.pkg.dev/all-in-demo/cloud-run-source-deploy/cognigy-gemini-speech:latest}"
APP_PORT="8080"
NETWORK_NAME="web_network"

# GCP Services configuration
PROJECT_ID="${PROJECT_ID:-tmo-circlek}"
LOCATION="${LOCATION:-europe-west1}"
TRANSCRIBE_LOCATION="${TRANSCRIBE_LOCATION:-eu}"
STT_PROVIDER="${STT_PROVIDER:-google}"
GOOGLE_SPEECH_LOCATION="${GOOGLE_SPEECH_LOCATION:-eu}"
GOOGLE_SPEECH_MODEL="${GOOGLE_SPEECH_MODEL:-telephony}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "====================================================================="
echo "Starting Cognigy Gemini Speech deployment..."
echo "Target image: ${APP_IMAGE}"
echo "Project:      ${PROJECT_ID} (STT provider: ${STT_PROVIDER})"
echo "====================================================================="

# ---------------------------------------------------------------------
# AUTOMATIC DOMAIN DETECTION (GCE Metadata -> sslip.io fallback)
# ---------------------------------------------------------------------
PUBLIC_IP=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" 2>/dev/null || true)

if [ -z "$PUBLIC_IP" ] || [[ "$PUBLIC_IP" == *"html"* ]]; then
    PUBLIC_IP=$(curl -s https://ifconfig.me || true)
fi

if [ -z "$PUBLIC_IP" ]; then
    echo "ERROR: Unable to detect external public IP." >&2
    exit 1
fi

DOMAIN="${PUBLIC_IP}.sslip.io"
echo "Public Domain: https://${DOMAIN}/"

# ---------------------------------------------------------------------
# CADDY REVERSE PROXY CONFIGURATION
# ---------------------------------------------------------------------
mkdir -p /var/lib/caddy/data
mkdir -p /var/lib/caddy/config

cat << EOF > /var/lib/caddy/Caddyfile
$DOMAIN {
    reverse_proxy my-application:$APP_PORT {
        # Streaming flush for ultra-low latency TTS chunked transfer
        flush_interval -1
    }
}
EOF

# ---------------------------------------------------------------------
# ARTIFACT REGISTRY AUTHENTICATION (Writeable /var/lib Fix)
# ---------------------------------------------------------------------
WRITABLE_HOME="/var/lib/docker-config"
mkdir -p "$WRITABLE_HOME/.docker"

if command -v docker-credential-gcr >/dev/null 2>&1; then
    HOME="$WRITABLE_HOME" docker-credential-gcr configure-docker --registries="${REGISTRY_REGION}-docker.pkg.dev" >/dev/null 2>&1 || true
fi

DOCKER_CMD="docker --config $WRITABLE_HOME/.docker"

# ---------------------------------------------------------------------
# DOCKER NETWORK SETUP (IDEMPOTENT)
# ---------------------------------------------------------------------
# Check if the bridge network already exists before trying to create it
if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    echo "Creating Docker network '${NETWORK_NAME}'..."
    docker network create "$NETWORK_NAME"
else
    echo "Docker network '${NETWORK_NAME}' already exists. Reusing existing network."
fi

# ---------------------------------------------------------------------
# FORCE PULL LATEST IMAGE (Bypasses Docker local tag cache)
# ---------------------------------------------------------------------
echo "Pulling latest container image from Artifact Registry..."
$DOCKER_CMD pull "$APP_IMAGE"

# ---------------------------------------------------------------------
# CLEAN CONTAINER REPLACEMENT (Prevents ghost endpoint collisions)
# ---------------------------------------------------------------------
echo "Stopping previous application container (if any)..."
# Force-disconnect from network first to avoid "endpoint already exists" daemon race
docker network disconnect -f "$NETWORK_NAME" my-application >/dev/null 2>&1 || true
docker rm -f my-application >/dev/null 2>&1 || true

# ---------------------------------------------------------------------
# START APPLICATION CONTAINER
# ---------------------------------------------------------------------
echo "Starting application container '${APP_IMAGE}'..."
$DOCKER_CMD run -d \
  --name my-application \
  --network "$NETWORK_NAME" \
  --restart always \
  -e PROJECT_ID="$PROJECT_ID" \
  -e LOCATION="$LOCATION" \
  -e TRANSCRIBE_LOCATION="$TRANSCRIBE_LOCATION" \
  -e STT_PROVIDER="$STT_PROVIDER" \
  -e GOOGLE_SPEECH_LOCATION="$GOOGLE_SPEECH_LOCATION" \
  -e GOOGLE_SPEECH_MODEL="$GOOGLE_SPEECH_MODEL" \
  -e LOG_LEVEL="$LOG_LEVEL" \
  "$APP_IMAGE"

# ---------------------------------------------------------------------
# MANAGE CADDY REVERSE PROXY CONTAINER
# ---------------------------------------------------------------------
if [ ! "$(docker ps -q -f name=caddy)" ]; then
    echo "Starting Caddy container..."
    docker rm -f caddy >/dev/null 2>&1 || true
    docker run -d \
      --name caddy \
      --network "$NETWORK_NAME" \
      -p 80:80 \
      -p 443:443 \
      -p 443:443/udp \
      -v /var/lib/caddy/Caddyfile:/etc/caddy/Caddyfile \
      -v /var/lib/caddy/data:/data \
      -v /var/lib/caddy/config:/config \
      --restart always \
      caddy:latest
else
    echo "Caddy is already running. Reloading configuration without dropping SSL..."
    docker exec caddy caddy reload --config /etc/caddy/Caddyfile >/dev/null 2>&1 || true
fi

echo "====================================================================="
echo "Deployment completed successfully!"
echo "Service is live at: https://${DOMAIN}/healthz"
echo "====================================================================="
