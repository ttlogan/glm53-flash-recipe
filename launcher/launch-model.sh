#!/usr/bin/env bash
# TURNKEY (on-head): launch the GX10 2-node vLLM serving stack AND update the head's /etc/motd.
# Runs ON the head node (rank 0 locally, rank 1 via ssh to the worker). The head holds its
# own key (~/.ssh/id_ed25519) whose pubkey is in the worker's authorized_keys.
# HEAD-FIRST is required: rank 0 owns the TCPStore (:29521); worker-first hits a
# Gloo "Connection reset by peer" NCCL init race.
# Call: launch-model.sh <glm|qwen|deepseek>
set -euo pipefail

MODEL="${1:?usage: launch-model.sh <glm|qwen|deepseek>}"
case "$MODEL" in glm|qwen|deepseek) ;; *) echo "unknown model '$MODEL'" >&2; exit 2 ;; esac

LEO=192.168.0.226
RAPH=192.168.0.143
# User owning the rootless podman containers + the ssh to the worker.
RUNTIME_USER="${RUNTIME_USER:-vllm}"
VLLM_SCRIPT="$HOME/glm53-vllm.sh"
# head's own key -> worker. Array form so ssh + flags are separate argv entries.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$HOME/.ssh/id_ed25519")
VLLM="bash $VLLM_SCRIPT"

# Vetted guard: confirm the model weights dir exists before relaunching a stack.
case "$MODEL" in
  glm)        MDIR=/models/glm-5.3-flash-nvfp4 ;;
  qwen)       MDIR=/models/qwen3.8-flash-next-uncensored-nvfp4-fp8ple ;;
  deepseek)   MDIR=/models/deepseek-v4-flash-vision-exp-ablit-nvfp4 ;;
esac
[ -f "$MDIR/config.json" ] || { echo "MODEL MISSING: $MDIR" >&2; exit 4; }

echo "==> [1/6] update head /etc/motd for model '$MODEL'"
sudo bash /opt/glm53/update-motd-model.sh "$MODEL" >/dev/null

echo "==> [2/6] stopping any existing serving containers (both nodes)"
podman rm -f vllm_glm53 2>/dev/null || true
"${SSH[@]}" "$RUNTIME_USER@$RAPH" 'podman rm -f vllm_glm53 2>/dev/null || true'

echo "==> [3/6] drop caches + launch HEAD (rank 0) on the head node"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
$VLLM 0 >/dev/null

echo "-- waiting for head TCPStore :29521"
for i in $(seq 1 30); do
  ss -ltnp 2>/dev/null | grep -q ":29521" && { echo "   TCPStore up"; break; }
  sleep 2
done

echo "==> [4/6] drop caches + launch WORKER (rank 1) on the worker node"
"${SSH[@]}" "$RUNTIME_USER@$RAPH" "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; bash $VLLM_SCRIPT 1" >/dev/null

echo "==> [5/6] waiting for /health on the head node :8000 (slow load ~10-15 min)"
for i in $(seq 1 90); do
  sleep 20
  c=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null)
  [ "$c" = "200" ] && { echo "   READY after ~$((i*20))s. Model: $MODEL"; exit 0; }
  [ $((i % 9)) -eq 0 ] && echo "   still loading ($((i*20))s)..."
done
echo "   TIMEOUT. Check: podman logs vllm_glm53 (head node)" >&2
exit 1
