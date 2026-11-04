#!/usr/bin/env bash
# 在单台 GPU 服务器上裸机启动同步 Number Search Tinker server（用于 p2a / p2b / p41）。
#
# 等价于 `uv run modal deploy serve_number_search_sync.py`，但不依赖 Modal：
# 复现 skyrl_server.py 容器镜像的构建步骤（pinned SkyRL fork + uv sync），
# 再用与 launch_tinker_server() 相同的命令启动 skyrl.tinker.api。
#
# 硬件要求：Ampere (sm80) 及以上 GPU；NVIDIA 驱动 >= 570（CUDA 12.8 运行时）。
# 所有 GPU 组件（vLLM / flash-attn / TransformerEngine / Megatron）均为预编译
# wheel，无需安装 CUDA toolkit。默认 2 卡 colocated（与 Modal 部署一致）；
# 单张 80GB 卡可用 SKYRL_NUM_GPUS=1（吞吐约减半，p2a 总量小，可以接受）。
#
# 用法（在本目录下）：
#   ./run_baremetal_p2a.sh                       # 默认 2 卡 colocated
#   SKYRL_NUM_GPUS=1 ./run_baremetal_p2a.sh      # 单卡 80GB
#   FORCE_SETUP=1 ./run_baremetal_p2a.sh         # 强制重新 clone + uv sync
#   Ctrl+C 停止服务器。
#
# 路径可用环境变量覆盖（默认值需要 root 权限写 /opt、/cache、/state）：
#   SKYRL_REPO_ROOT   SkyRL 仓库位置    默认 /opt/SkyRL
#   SKYRL_VENV        venv 位置         默认 /opt/skyrl-venv
#   HF_HOME           HF 缓存           默认 /cache/huggingface
#   SKYRL_STATE_DIR   checkpoint/数据库 默认 /state
#   SKYRL_BASE_MODEL  服务的基座模型    默认 Qwen/Qwen3.5-4B
#   SKYRL_NUM_GPUS    GPU 数（1 或 2）  默认 2
#
# 服务器就绪后（首次启动要下载模型、拉起 Ray/vLLM，需数分钟）：
#   export TINKER_BASE_URL=http://127.0.0.1:9000    # 训练脚本在同一台机器
#   # 或 http://<服务器IP>:9000（跨机时记得放行端口），然后在仓库根目录：
#   uv run python scripts/train_sync.py --config configs/p2a_number_search_reinforce.yaml

set -euo pipefail

# ---- 常量（与 skyrl_server.py 一致） ----
SKYRL_GIT_URL="https://github.com/adityasoni9998/SkyRL.git"
SKYRL_COMMIT="491e1329c314ef43818fb6824d9f2ae61883416c"
BACKEND="megatron"
REMOTE_REPO="${SKYRL_REPO_ROOT:-/opt/SkyRL}"
VENV="${SKYRL_VENV:-/opt/skyrl-venv}"
HF_HOME="${HF_HOME:-/cache/huggingface}"
STATE_DIR="${SKYRL_STATE_DIR:-/state}"
MODEL_NAME="${SKYRL_BASE_MODEL:-Qwen/Qwen3.5-4B}"
PORT=9000
NUM_GPUS="${SKYRL_NUM_GPUS:-2}"

# ---- 0. 前置检查 ----
command -v git >/dev/null || { echo "缺少 git" >&2; exit 1; }
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  echo "缺少 uv：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || {
  echo "未找到 nvidia-smi，请先安装 NVIDIA 驱动 (>= 570)" >&2
  exit 1
}
gpu_count="$(nvidia-smi --list-gpus | wc -l)"
(( gpu_count >= NUM_GPUS )) || {
  echo "本机只有 $gpu_count 块 GPU，backend_config 要求 $NUM_GPUS 块" >&2
  exit 1
}

export PATH="$HOME/.local/bin:$VENV/bin:$PATH"

# ---- 1. 获取 pinned SkyRL fork（同镜像的 _clone_skyrl，含 commit 校验） ----
current_commit="$(git -C "$REMOTE_REPO" rev-parse HEAD 2>/dev/null || true)"
if [[ "${FORCE_SETUP:-0}" == 1 || "$current_commit" != "$SKYRL_COMMIT" ]]; then
  echo ">>> 获取 SkyRL @ $SKYRL_COMMIT ..."
  git init -q "$REMOTE_REPO" 2>/dev/null || true
  git -C "$REMOTE_REPO" remote remove origin 2>/dev/null || true
  git -C "$REMOTE_REPO" remote add origin "$SKYRL_GIT_URL"
  git -C "$REMOTE_REPO" fetch --depth=1 origin "$SKYRL_COMMIT"
  git -C "$REMOTE_REPO" checkout --detach FETCH_HEAD
  test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$SKYRL_COMMIT"
else
  echo ">>> SkyRL @ $SKYRL_COMMIT 已就绪"
fi

# ---- 2. 安装依赖（同镜像的 uv sync + 导入检查） ----
if [[ "${FORCE_SETUP:-0}" == 1 || ! -f "$VENV/.setup-ok" ]]; then
  echo ">>> uv sync --extra tinker --extra $BACKEND（首次需下载数 GB）..."
  (
    cd "$REMOTE_REPO" &&
      UV_PROJECT_ENVIRONMENT="$VENV" UV_LINK_MODE=copy \
        uv sync --extra tinker --extra "$BACKEND"
  )
  # 与镜像的 _IMPORT_CHECK 相同：装坏的依赖在这里失败，而不是第一次训练时
  "$VENV/bin/python" -c 'import skyrl; import torch, vllm, flash_attn, transformer_engine.pytorch, megatron.core, megatron.bridge; assert torch.cuda.is_available(); print(torch.__version__, vllm.__version__, transformer_engine.__version__)'
  touch "$VENV/.setup-ok"
fi

# ---- 3. p2a 的 backend_config ----
# = skyrl_server.py 的 BASE_BACKEND_CONFIG + colocated(2) 的三个 placement 键
#   + serve_number_search_sync.py 的四个覆盖项。单卡跑法无需改本文件，
#   用 SKYRL_NUM_GPUS=1 即可，占位符 __NUM_GPUS__ 会同时替换两处。
BACKEND_CONFIG="$(cat <<'JSON'
{
  "trainer.placement.policy_num_nodes": 1,
  "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
  "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
  "trainer.policy.megatron_config.context_parallel_size": 1,
  "trainer.policy.language_model_only": true,
  "generator.inference_engine.language_model_only": true,
  "trainer.policy.megatron_config.lora_config.merge_lora": false,
  "trainer.policy.optimizer_config.max_grad_norm": 1.0,
  "trainer.policy.optimizer_config.adam_betas": [0.9, 0.95],
  "trainer.policy.optimizer_config.weight_decay": 0.0,
  "trainer.use_expandable_segments": true,
  "trainer.max_tokens_per_microbatch": 32768,
  "generator.inference_engine.max_num_batched_tokens": 50000,
  "generator.inference_engine.max_num_seqs": 256,
  "generator.inference_engine.enable_prefix_caching": true,
  "generator.inference_engine.enforce_eager": false,
  "generator.inference_engine.engine_init_kwargs.gdn_prefill_backend": "triton",
  "generator.inference_engine.engine_init_kwargs.compilation_config": {
    "cudagraph_mode": "FULL_DECODE_ONLY"
  },
  "generator.inference_engine.use_expandable_segments": false,
  "generator.inference_engine.enable_ray_prometheus_stats": true,
  "generator.inference_engine.engine_init_kwargs.enable_mfu_metrics": true,
  "trainer.placement.colocate_all": true,
  "trainer.placement.policy_num_gpus_per_node": __NUM_GPUS__,
  "generator.inference_engine.num_engines": __NUM_GPUS__,
  "trainer.logprobs_chunk_size": 1024,
  "trainer.fused_lm_head_logprob": true,
  "generator.inference_engine.gpu_memory_utilization": 0.7,
  "generator.inference_engine.engine_init_kwargs.max_model_len": 32768
}
JSON
)"
BACKEND_CONFIG="${BACKEND_CONFIG//__NUM_GPUS__/$NUM_GPUS}"

# ---- 4. 启动（环境变量与 launch_tinker_server() / 镜像 .env 一致） ----
mkdir -p "$STATE_DIR/checkpoints" "$STATE_DIR/database"
[[ -d /usr/local/cuda ]] && export CUDA_HOME=/usr/local/cuda
export HF_HOME
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_LINK_MODE=copy
export SKYRL_REPO_ROOT="$REMOTE_REPO"
export RAY_RUNTIME_ENV_HOOK="ray._private.runtime_env.uv_runtime_env_hook.hook"
# hf_transfer 已安装才启用加速下载，避免缺包时报错
if "$VENV/bin/python" -c 'import hf_transfer' 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
# 同 launch_tinker_server()：Ray/SkyRL 以 /tmp 为 HOME，HF 缓存仍走 HF_HOME
export HOME=/tmp

echo ">>> serving base model: $MODEL_NAME"
echo ">>> GPUs: $NUM_GPUS colocated（训练与 vLLM 采样同卡分相）"
echo ">>> 就绪后设置: export TINKER_BASE_URL=http://<本机IP>:$PORT"
echo ">>> 按 Ctrl+C 停止服务器"

cd "$REMOTE_REPO"
export VIRTUAL_ENV="$VENV"
exec uv run --active --no-sync --extra tinker --extra "$BACKEND" \
  -m skyrl.tinker.api \
  --base-model "$MODEL_NAME" \
  --backend "$BACKEND" \
  --port "$PORT" \
  --checkpoints-base "$STATE_DIR/checkpoints" \
  --database-url "sqlite:///$STATE_DIR/database/tinker.db" \
  --backend-config "$BACKEND_CONFIG"
