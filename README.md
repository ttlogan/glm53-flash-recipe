# GLM-5.3-Flash NVFP4 — 2-node GX10 Serve Recipe

Production recipe for serving **`GLM-5.3-Flash NVFP4`** (`glm5_next`) across a
**2-node NVIDIA GB10 / DGX Spark cluster** with **tensor-parallel 2**, an
**SM121 top-k (kpool) fix overlay**, optional **DFlash2 speculative decoding**,
and the **launch-orchestration fixes** that make it idempotent across reboots.

This repo is the **sanitized configuration + documentation** for a working
setup. Real usernames and all secrets are removed; the runtime user is
configurable. Internal IPs are preserved (they describe the cluster fabric), but
real hostnames are replaced with `head`/`worker`.

---

## What's in this repo

| Path | What it is | Source |
|---|---|---|
| `launcher/glm53-vllm-ranklocal.sh` | **No-drafter lane**: GLM-5.3-Flash NVFP4, vLLM TP=2, runs rank-local (arg `0`=head, `1`=worker). No-drafter `sm121-v8` image. | Own work (adapted from tonyd2wild) |
| `launcher/glm53-vllm-dflash2-ranklocal.sh` | **DFlash2 speculative lane**: same model + DFlash2 draft, `sm121-v11-dflash2` image. | Own work (adapted from tonyd2wild) |
| `launcher/launch-model.sh` | **TURNKEY swap/launch**: update motd + HEAD-first launch of `glm\|qwen\|deepseek`. | Own work |
| `launcher/glm53-vllm.service` | systemd oneshot auto-start for the GLM vLLM head (rank 0). | Own work |
| `scripts/update-motd-model.sh` | Regenerate `/etc/motd` tagging the active model + launch flags. | Own work |
| `scripts/default_model_vllm.sh` | **Default-model boot/ensure launcher** for the cluster's recipe stack. | Own work |
| `patches/sparse_attn_indexer_kpool.py` | **SM121 top-k fix overlay** — without it the engine dies past ~24K ctx. | vLLM/vendor source (see Attribution) |

---

## The model

- **`GLM-5.3-Flash NVFP4`** (`glm5_next`) — GLM 5.3 Flash, NVFP4 weights,
  multimodal (vision), 1M context. Served as `dgx_hobo_default`.
  **HF repo:** [https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (NVIDIA ModelOpt quant of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash))
- Loaded via the **tonyd2wild `vllm-glm53-flash`** container image (GB10/sm_121
  build), in two lanes:
  - **no-drafter**: `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8`
  - **DFlash2 speculative**: `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
- **SM121 top-k fix overlay** (`patches/sparse_attn_indexer_kpool.py`) must be
  mounted into the container at
  `.../vllm/model_executor/layers/sparse_attn_indexer_kpool.py`. Without it the
  engine dies past ~24K context.

---

## Cluster topology

| Role | Host (real LAN IP) | RoCE IP | Rank |
|---|---|---|---|
| Head (server) | 192.168.0.226 | 192.168.177.11 | 0 |
| Worker | 192.168.0.143 | 192.168.177.12 | 1 |

- 2 nodes, TP=2, API exposed on the **head** only (`:8000`), fronted by LiteLLM
  at `:4000` with a single fixed label `dgx_hobo_default`.
- RoCE fabric: `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_GID_INDEX=3`,
  `NCCL_IB_ADDR_RANGE=192.168.177.0/24`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`.
- `--nnodes 2` with `--master-addr <head RoCE>` `--master-port 29521`
  (`--distributed-executor-backend mp`).

---

## Tuning (the GLM5.3 launch flags)

The GLM5.3-Flash NVFP4 serve flags (from `update-motd-model.sh`):

```
--tensor-parallel-size 2 --nnodes 2 --moe-backend marlin
--kv-cache-dtype fp8_e4m3 --kv-cache-memory 8589934592
--max-model-len 1048576 --gpu-memory-utilization 0.85
--max-num-seqs 6 --block-size 2304 --max-num-batched-tokens 8192
--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice
--enforce-eager
--default-chat-template-kwargs '{"enable_thinking":false}'
--chat-template <model>/chat_template.jinja
--served-model-name dgx_hobo_default
```

- **`block-size 2304`** is the GLM5.3 sparse-attention pool granularity; do not
  change it independently of the model.
- **`kv-cache-dtype fp8_e4m3` + `kv-cache-memory 8589934592` (8 GiB)** cap the
  KV pool; `--gpu-memory-utilization 0.85` reserves the rest.
- **DFlash2 lane** additionally passes:
  ```
  --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}'
  ```

---

## The SM121 top-k fix (why `patches/`)

On GB10/sm_121 the GLM5.3 sparse-attention indexer's `kpool_compress` path
crashes / dies past ~24K context without a patched
`sparse_attn_indexer_kpool.py`. The fix overlay is mounted read-only into the
container:

```sh
-v "$HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro"
```

Both rank-local launchers **require** `$HOME/patches/sparse_attn_indexer_kpool.py`
to exist (they `exit 3` if missing). Fetch it to `~/patches/` before launching.

---

## Launch orchestration (the non-obvious parts)

### 1. HEAD-first (rank 0 before rank 1)
Rank 0 owns the TCPStore on `:29521`. Launching the worker first hits a
Gloo `Connection reset by peer` NCCL init race. `launch-model.sh` launches the
**head first**, waits for the TCPStore to answer (`ss -ltnp | grep :29521`),
then launches the worker. This ordering is mandatory.

### 2. `--network host` is required for NCCL/RoCE
This is a **rootful multi-node NCCL/RoCE pod**. NCCL must own the NIC and see
`rocep*` directly, so the container runs with `--network host --ipc host`
(not pasta, not bridge). `--device /dev/infiniband` is passed through.

### 3. Drop caches before each launch
Both nodes run `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before
launching, to avoid cold-start OOM on the ~185G model.

### 4. Runtime user is configurable
The real deployment runs the rootless containers as a dedicated service user.
In this repo, every `sudo -u <user>` / `ssh <user>@` is driven by
`RUNTIME_USER` (default `vllm`). Set it to your real user at deploy time.

---

## Boot autostart

`launcher/glm53-vllm.service` (systemd oneshot, on the head node) starts the
GLM vLLM head at boot:

```ini
[Unit]
Description=GLM-5.3-Flash NVFP4 vLLM head (rank 0, TP=2)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=vllm
ExecStartPre=/bin/sleep 30
ExecStart=/home/vllm/glm53-vllm.sh 0
RemainAfterExit=yes
Restart=on-failure
RestartSec=30
TimeoutStartSec=2400
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

`scripts/default_model_vllm.sh` is the higher-level **default-model ensure**
script (idempotent; updates motd; `--swap <qwen|glm|deepseek>`). It drives the
recipe stack for `deepseek` and delegates GLM/Qwen to `launch-model.sh`.

---

## Reproduce (docs)

1. On the head node, place `glm53-vllm-ranklocal.sh` / `glm53-vllm-dflash2-ranklocal.sh`
   at `~/glm53-vllm.sh` (and `~/glm53-vllm-dflash2.sh` respectively).
2. Ensure `/models/glm-5.3-flash-nvfp4` (and `/models/dflash2-draft` for the
   DFlash2 lane) exist on both nodes.
3. Fetch `patches/sparse_attn_indexer_kpool.py` to `~/patches/`.
4. Set `RUNTIME_USER` to the account that owns the rootless containers.
5. For the no-drafter lane:
   ```sh
   # on the head node, as the runtime user
   bash ~/glm53-vllm.sh 0        # rank 0 (head)
   ssh <runtime-user>@<worker> 'bash ~/glm53-vllm.sh 1'   # rank 1 (worker)
   ```
   Or TURNKEY via `launch-model.sh glm`.
6. Wait for `/health` → 200 on the head `:8000`, then `curl :4000/v1/models`
   (LiteLLM) to confirm `dgx_hobo_default`.

---

## Attribution / provenance

- **Container image** `ghcr.io/tonyd2wild/vllm-glm53-flash` — the GB10/sm_121
  vLLM GLM5.3 build; the `-dflash2` tag adds DFlash2 speculative decoding.
- **Launch scripts** adapted from tonyd2wild's `launch-glm53-vllm-tp2.sh` and
  the DFlash2 docs, for podman + this cluster's RoCE fabric.
- **`patches/sparse_attn_indexer_kpool.py`** is vLLM source (Apache-2.0) with an
  SM121-aware kpool overlay fix.
- **Model** GLM-5.3-Flash NVFP4 (`glm5_next`) — the upstream GLM release.

> No secrets, tokens, or real usernames are present. Internal IPs are documented
> because they describe the cluster fabric; replace them if your fabric differs.
> Set `RUNTIME_USER` at deploy time.

---
