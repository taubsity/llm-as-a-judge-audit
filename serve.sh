#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

serve() {
    local port=$1 tp=$2
    shift 2
    docker run --runtime nvidia --gpus all --ipc=host \
        -p "$port":8000 \
        -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
        -e HF_TOKEN \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e OMP_NUM_THREADS=4 \
        -e VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
        vllm/vllm-openai:v0.18.0 \
        "$@" \
        --disable-custom-all-reduce \
        --trust-remote-code \
        --tensor-parallel-size "$tp" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 100000 \
        --reasoning-parser deepseek_r1
}

case "${1:?usage: serve.sh <Q-235B|Q-80B|MG-27B|G-120B|M-24B|Q-235B-Think>}" in
    Q-235B) serve 8001 4 --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 ;;
    Q-80B) serve 8002 1 --model Qwen/Qwen3-Next-80B-A3B-Instruct --served-model-name Qwen3-Next-80B-A3B-Instruct ;;
    MG-27B) serve 8003 1 --model google/medgemma-27b-it ;;
    G-120B) serve 8004 1 --model openai/gpt-oss-120b ;;
    M-24B) serve 8005 1 --model mistralai/Magistral-Small-2509 ;;
    Q-235B-Think) serve 8006 4 --model Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 ;;
    *)
        echo "unknown model: $1" >&2
        exit 1
        ;;
esac
