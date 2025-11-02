# HumbleAILLMs - Multi-GPU Evaluation System

# Simple-Evals Setup Guide

## Installation

```bash
# Clone repository
git clone https://github.com/sebasmos/HumbleAILLMs.git

# Install dependencies
pip install tabulate optimum gptqmodel auto-gptq torch transformers accelerate openai anthropic human-eval bitsandbytes autoawq auto-gptq

# Optional: Set API keys for cloud models (not needed for local models)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Optional: Set HuggingFace token for gated models (MedGemma, etc.)
export HF_TOKEN="your-hf-token-here"  # Get from https://huggingface.co/settings/tokens
```

## Download Recommended Models (Pre-Quantized for Efficiency)

**IMPORTANT**: The default grader in `simple_evals.py` is `Qwen/Qwen2.5-14B-Instruct-AWQ` (~7GB VRAM). Install it before you launch evaluations:

```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ
```

**Evaluation model suggestions (pick the tier that matches your GPU)**

- **12–16GB GPUs** (RTX 3080, RTX 4070 Ti, L40s)
    - `huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ`
    - Optional: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ`
- **16–24GB GPUs** (RTX 3090, RTX 4080)
    - `huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ`
    - Optional: `huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4`
- **24–48GB GPUs** (RTX 4090, L40, A100 40GB)
    - `huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ`
    - Optional: `huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- **96GB+ or multi-GPU nodes**
    - `huggingface-cli download meta-llama/Meta-Llama-3-70B`
    - `huggingface-cli download meta-llama/Llama-3.3-70B-Instruct`
    - `huggingface-cli download Qwen/Qwen2-57B-A14B-Instruct`
    - `huggingface-cli download Qwen/Qwen2-72B-Instruct`

> Pre-downloading keeps HealthBench runs from stalling while weights stream in. Models still download lazily on first use if you skip this step.

## Quick Test

```bash
# List available models
python -m simple-evals.simple_evals --list-models

# Test with API model
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=mmlu --examples=1

# Test with local model
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=mmlu --examples=10
```

## Available Local Models

```bash
# High-end FP8 / multi-GPU (≥96GB or tensor parallel)
Llama-3.1-405B-Instruct-FP8
Llama-3.1-405B-FP8
meta-llama/Meta-Llama-3-70B
meta-llama/Llama-3.3-70B-Instruct
Qwen/Qwen2-72B-Instruct
Qwen/Qwen2-57B-A14B-Instruct
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

# Large research models (≥40GB VRAM recommended)
mistralai/Mistral-Large-Instruct-2407
gpt-oss-120b              # Use --quantize 4bit or multi-GPU
gpt-oss-20b
qwen3-32b
qwen3-30b-a3b
deepseek-r1-qwen-32b

# Lightweight general models
gpt-neo-1.3b

# Medical models (HF_TOKEN required)
medgemma-4b-it
medgemma-4b-pt
medgemma-27b-it
medgemma-27b-text-it

# Qwen pre-quantized (fastest path for HealthBench)
qwen2.5-3b-instruct-awq
qwen2.5-7b-instruct-awq
qwen2.5-7b-instruct-gptq
qwen2.5-14b-instruct-awq  # Default grader
qwen2.5-14b-instruct-gptq-int4

# Full precision Qwen backbones
qwen2.5-14b-instruct
qwen2.5-14b

# Dynamic 4-bit loaders (extra RAM during init)
qwen2.5-14b-instruct-4bit
qwen2.5-14b-4bit
```

## HealthBench Evaluations

```bash
# Fast sanity check (fits 12GB GPUs)
python -m simple-evals.simple_evals --model=qwen2.5-3b-instruct-awq --eval=healthbench_hard --examples=5

# Recommended baseline (matches default grader)
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct-awq --eval=healthbench_hard --examples=10

# High-capacity single node (set --quantize 4bit if memory is tight)
python -m simple-evals.simple_evals --model=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --eval=healthbench_hard --examples=10 --quantize 4bit

# Multi-GPU / tensor-parallel runs
python -m simple-evals.simple_evals --model=meta-llama/Llama-3.3-70B-Instruct --eval=healthbench_hard --examples=10 --quantize 4bit
python -m simple-evals.simple_evals --model=Qwen/Qwen2-72B-Instruct --eval=healthbench_hard --examples=10 --quantize 4bit

# All HealthBench variants in one go
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct-awq --eval=healthbench,healthbench_hard,healthbench_consensus --examples=10
```

## Key Parameters

- `--examples=N` → Limit to N samples per evaluation
- `--debug` → Use minimal samples + verbose output
- `--n-threads=N` → Control parallelization (default: 120)
- `--model=name1,name2` → Test multiple models
- `--eval=eval1,eval2` → Run multiple evaluations

## Output

Results saved to `/tmp/` as:
- HTML reports: `{eval}_{model}_{timestamp}.html`
- JSON metrics: `{eval}_{model}_{timestamp}.json`
Complete system for running LLM evaluations on MIT Engaging cluster with automatic multi-GPU support.

---