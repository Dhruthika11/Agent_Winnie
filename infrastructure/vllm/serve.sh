#!/usr/bin/env bash
# Launches vLLM's OpenAI-compatible server for the configured Hermes-style
# instruct model. --tool-call-parser hermes enables native parsing of
# Hermes/NousResearch-style function-call tokens; swap the parser if you
# point this at a Qwen or Mistral checkpoint instead.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-NousResearch/Hermes-3-Llama-3.1-8B}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

exec vllm serve "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --served-model-name "${MODEL_NAME}"
