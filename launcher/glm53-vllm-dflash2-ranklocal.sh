#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 (glm5_next) + DFlash2 speculative decoding, vLLM TP=2 on 2x GX10 (GB10/sm_121).
# DFlash2 lane (tonyd2wild sm121-v11-dflash2 image). Worker (rank 1) FIRST, then head (rank 0).
# Adapted from glm53-vllm.sh per tonyd2wild CURRENT.md (2026-09-29) + DFLASH2-SPECULATIVE-DECODING.md.
set -euo pipefail

NODE_RANK="${1:?usage: glm53-vllm-dflash2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2"
NAME="vllm_glm53_dflash2"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
DRAFTER_PATH="/models/dflash2-draft"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
# cluster fabric: head=.11 (rank0), worker=.12 (rank1); master addr = head
HEAD_IP="192.168.177.11"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.177.11; HEADLESS="" ;;
  1) HOST_IP=192.168.177.12; HEADLESS="--headless" ;;
esac

test -f "$MODEL_PATH/config.json" || { echo "MODEL MISSING: $MODEL_PATH" >&2; exit 4; }
test -f "$DRAFTER_PATH/model.safetensors" || { echo "DRAFTER MISSING: $DRAFTER_PATH" >&2; exit 5; }
mkdir -p "$CACHE_HOST_PATH"

# SM121 top-k fix overlay (fetch to ~/patches first). Without it engine dies past ~24K ctx.
[ -f "$HOME/patches/sparse_attn_indexer_kpool.py" ] || { echo "missing kpool patch" >&2; exit 3; }

podman rm -f "$NAME" 2>/dev/null || true

podman run -d \
  --name "$NAME" --restart no \
  --device nvidia.com/gpu=all \
  --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_PATH:$MODEL_PATH:ro" \
  -v "$DRAFTER_PATH:$DRAFTER_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=192.168.177.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -v "$HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name dgx_hobo_default \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-seqs 6 --block-size 2304 --moe-backend marlin \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory 8589934592 \
    --max-num-batched-tokens 8192 \
    --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}' \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking":false}' \
    --chat-template "$MODEL_PATH/chat_template.jinja" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"
sleep 2
podman ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited; inspect with: podman logs $NAME" >&2; exit 1; }
