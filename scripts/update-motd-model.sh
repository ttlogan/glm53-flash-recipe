#!/usr/bin/env bash
# Regenerate /etc/motd for the GX10 cluster head node, tagging the currently
# active model with its launch flags. Call: sudo update-motd-model.sh <model>
#   <model> in: glm | qwen | deepseek   (glm = default)
#   --list : show available models
#   --active : print which model the motd currently advertises
set -euo pipefail

MODEL_BASE="/models"

declare -A NAME VISION CTX SIZE ROLE FLAGS
NAME[glm]="GLM-5.3-Flash NVFP4"
VISION[glm]="yes"
CTX[glm]="1M"
SIZE[glm]="185G"
ROLE[glm]="multimodal (default)"
FLAGS[glm]="--tensor-parallel-size 2 --nnodes 2 --moe-backend marlin --kv-cache-dtype fp8_e4m3 --max-model-len 1048576 --gpu-memory-utilization 0.85 --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --enforce-eager --served-model-name dgx_hobo_default"

NAME[qwen]="Qwen3.8-Flash-Next NVFP4/FP8"
VISION[qwen]="yes"
CTX[qwen]="262K"
SIZE[qwen]="124G"
ROLE[qwen]="vision (fast, tool-calling)"
FLAGS[qwen]="--tensor-parallel-size 2 --nnodes 2 --enable-expert-parallel --all2all-backend allgather_reducescatter --speculative-config mtp --gpu-memory-utilization 0.835 --max-num-seqs 8 --max-num-batched-tokens 8192 --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-auto-tool-choice --max-model-len 262144 --served-model-name dgx_hobo_default"

NAME[deepseek]="DeepSeek-V4-Flash-Vision NVFP4 (ablit)"
VISION[deepseek]="yes"
CTX[deepseek]="1M"
SIZE[deepseek]="165G"
ROLE[deepseek]="multimodal (1M, DSpark)"
FLAGS[deepseek]="--tensor-parallel-size 2 --nnodes 2 --kv-cache-dtype nvfp4_ds_mla --block-size 256 --max-model-len 1048576 --gpu-memory-utilization 0.74 --speculative-config dspark --moe-backend flashinfer_b12x --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice --served-model-name dgx_hobo_default"


usage() { echo "usage: sudo $0 <glm|qwen|deepseek> | --list | --active" >&2; }

case "${1:-}" in
  --list)
    for m in glm qwen deepseek; do printf "  %-9s %-28s %s %s\n" "$m" "${NAME[$m]}" "${SIZE[$m]}" "${CTX[$m]}"; done
    exit 0 ;;
  --active)
    grep -E "Active model|model = " /etc/motd 2>/dev/null || echo "(none)"; exit 0 ;;
  "") usage; exit 2 ;;
esac

MODEL="$1"
[[ " ${!NAME[*]} " == *" $MODEL "* ]] || { echo "unknown model '$MODEL'" >&2; usage; exit 2; }

# confirm the checkpoint is present (per-model dir name)
case "$MODEL" in
  glm) PDIR="glm-5.3-flash-nvfp4" ;;
  qwen) PDIR="qwen3.8-flash-next-uncensored-nvfp4-fp8ple" ;;
  deepseek) PDIR="deepseek-v4-flash-vision-exp-ablit-nvfp4" ;;
esac
[ -f "$MODEL_BASE/$PDIR/config.json" ] || echo "WARN: $MODEL checkpoint not found ($MODEL_BASE/$PDIR)" >&2

cat > /etc/motd <<EOF

  GX10 2-node cluster -- HEAD node (rank 0 / front door)

  Access (single HTTPS front door, self-signed cert):
    https://head/                -> landing page (links to all services)
    https://head/sparkdash/      -> sparkDash dashboard   (admin / <password>)
    https://head/ups/            -> UPS web UI            (upsadmin / <password>)

  Serving front door (one fixed label dgx_hobo_default):
    curl http://127.0.0.1:4000/v1/models      # LiteLLM (forwards to vLLM :8000)
    model label = dgx_hobo_default            # FIXED, never changes

  Active model: ${NAME[$MODEL]}  (${ROLE[$MODEL]}, ${SIZE[$MODEL]}, ctx ${CTX[$MODEL]}, vision ${VISION[$MODEL]})
    launch flags: ${FLAGS[$MODEL]}

  Stack management (from the Hermes host that holds the cluster jump key):
    cd <hermes-host> && ./launch-model.sh glm|qwen|deepseek   # TURNKEY: swap model + relaunch (HEAD-first)
    ./launch-model.sh --health        # just check /health
    podman ps --filter name=vllm_glm53                               # serving containers on head/worker

  Model swap (updates this motd AND must relaunch the stack):
    On the Hermes host:  ./launch-model.sh <model>
    (This updates /etc/motd on the head node, then HEAD-first relaunches the serving stack.)


  NUT / UPS:   sudo upsc cyberpower@localhost | head
  Models:      /models (Qwen, DeepSeek, GLM)

EOF

echo "motd updated for model: $MODEL (${NAME[$MODEL]})"
