# Simple-Evals Setup Guide

## Installation

```bash
# Clone repository
git clone https://github.com/sebasmos/HumbleAILLMs.git

# Install dependencies
pip install torch transformers accelerate openai anthropic human-eval bitsandbytes autoawq auto-gptq

# Optional: Set API keys for cloud models (not needed for local models)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Optional: Set HuggingFace token for gated models (MedGemma, etc.)
export HF_TOKEN="your-hf-token-here"  # Get from https://huggingface.co/settings/tokens
```

## Download Recommended Models (Pre-Quantized for Efficiency)

**IMPORTANT**: The grader model is set to `Qwen/Qwen2.5-7B-Instruct-AWQ` by default (~4GB VRAM). Download it before running evaluations:

```bash
# Download grader model (required - ~4GB download)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ

# Download evaluation models (choose based on your GPU memory):

# For 12-16GB GPU (e.g., RTX 3080, RTX 4070 Ti, L40s)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ        # ~2GB VRAM (evaluation model)
# Total: 4GB (grader) + 2GB (eval) = 6GB

# For 16-24GB GPU (e.g., RTX 3090, RTX 4080)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ       # ~4GB VRAM (evaluation model)
# Total: 4GB (grader) + 4GB (eval) = 8GB

# For 24GB+ GPU (e.g., RTX 4090, L40, A100)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ       # ~7GB VRAM (evaluation model)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 # ~7GB VRAM (alternative)
# Total: 4GB (grader) + 7GB (eval) = 11GB
```

**Note**: If you don't pre-download, models will download automatically when first used (may appear to "hang" during download).

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
# General purpose models
gpt-neo-1.3b         # 1.3B parameter general model (~3GB VRAM)
gpt-oss-20b          # 20B parameter model (~40GB VRAM)

# Medical models (require HF_TOKEN and license acceptance)
medgemma-4b-it         # 4B instruction-tuned model (default MedGemma choice)
medgemma-4b-pt         # 4B base model for custom fine-tuning
medgemma-27b-it        # 27B instruction-tuned model (high accuracy, heavy VRAM)
medgemma-27b-text-it   # 27B instruction-tuned, text-only variant
                       # Accept licenses via Hugging Face:
                       # https://huggingface.co/google/medgemma-4b-it
                       # https://huggingface.co/google/medgemma-4b-pt
                       # https://huggingface.co/google/medgemma-27b-it
                       # https://huggingface.co/google/medgemma-27b-text-it

# Qwen Models - Pre-Quantized (RECOMMENDED - loads faster, less VRAM)
qwen2.5-3b-instruct-awq    # 3B instruct model (~2GB VRAM) ⭐ BEST FOR 6GB GPU
qwen2.5-7b-instruct-awq    # 7B instruct model (~4GB VRAM) ⭐ BEST FOR 8GB GPU
qwen2.5-7b-instruct-gptq   # 7B instruct GPTQ (~4GB VRAM) - Alternative to AWQ
qwen2.5-14b-instruct-awq   # 14B instruct AWQ (~7GB VRAM) ⭐ BEST FOR 12-16GB GPU
qwen2.5-14b-instruct-gptq  # 14B instruct GPTQ (~7GB VRAM) - Default grader model

# Full precision (FP16) variants - for reference
qwen2.5-14b-instruct      # MMLU 79.7%, 14.8B params (~28GB VRAM)
qwen2.5-14b               # Base model (~28GB VRAM)
qwen3-32b                 # MMLU 83-85%, exceptional performance (~64GB VRAM)
deepseek-r1-qwen-32b      # Beats o1-mini, reasoning-focused (~64GB VRAM)
qwen3-30b-a3b             # 3B active params, ArenaHard 91.0 (~60GB VRAM)

# Dynamic 4-bit quantization (quantizes on load - needs more RAM initially)
qwen2.5-14b-instruct-4bit # MMLU 79.7% (requires ~10-12GB to load, then ~7GB)
qwen2.5-14b-4bit          # Base model (requires ~10-12GB to load, then ~7GB)
```

## Basic Evaluations (Local Models)

```bash
# Multiple choice (no grader needed)
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=mmlu --examples=1 --debug
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=gpqa --examples=1 --debug

# Math and reasoning
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=mgsm --examples=5 --debug
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=drop --examples=5 --debug

# Medical evaluation with MedGemma (requires HF_TOKEN)
export HF_TOKEN="hf_your_token_here"
python -m simple-evals.simple_evals --model=medgemma-4b-it --eval=healthbench --examples=5
# Or try the larger variant
python -m simple-evals.simple_evals --model=medgemma-27b-it --eval=healthbench --examples=5
```

## HealthBench Evaluations

```bash
# Standard HealthBench
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=healthbench --examples=1

python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench --examples=10 

# Challenging subset
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=healthbench_hard --examples=5 --debug

# High-agreement subset  
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=healthbench_consensus --examples=10 --debug

# All HealthBench variants
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=healthbench,healthbench_hard,healthbench_consensus --examples=10 --debug
```

## Key Parameters

- `--examples=N` → Limit to N samples per evaluation
- `--debug` → Use minimal samples + verbose output
- `--n-threads=N` → Control parallelization (default: 120)
- `--model=name1,name2` → Test multiple models
- `--eval=eval1,eval2` → Run multiple evaluations

## GPU Memory Tier Configurations

The evaluation system uses **Qwen 2.5 7B Instruct AWQ (~4GB VRAM)** as the default grader for efficient grading.

### ⚠️ IMPORTANT: Grader Memory Requirements
- **Grading Model**: Qwen/Qwen2.5-7B-Instruct-AWQ (~4GB GPU RAM)
- **Available for Evaluation Model**: GPU_Total - 4GB

### Tier 1: 12-16GB GPU (e.g., RTX 3080, RTX 4070 Ti, L40s)
```bash
# Best choice: Qwen 2.5 3B Instruct AWQ (lightest)
python -m simple-evals.simple_evals --model=qwen2.5-3b-instruct-awq --eval=healthbench_hard --examples=10

# Alternative: Qwen 2.5 7B GPTQ (better quality)
python -m simple-evals.simple_evals --model=qwen2.5-7b-instruct-gptq --eval=healthbench_hard --examples=10
```
**Memory usage**: 4GB (grader) + 2-4GB (model) = ~6-8GB total

### Tier 2: 16-24GB GPU (e.g., RTX 3090, RTX 4080)
```bash
# Best choice: Qwen 2.5 7B Instruct AWQ
python -m simple-evals.simple_evals --model=qwen2.5-7b-instruct-awq --eval=healthbench_hard --examples=10

# Alternative: Qwen 2.5 14B AWQ (higher quality)
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct-awq --eval=healthbench_hard --examples=10
```
**Memory usage**: 4GB (grader) + 4-7GB (model) = ~8-11GB total

### Tier 3: 24-40GB GPU (e.g., RTX 4090, A10, A100 24GB)
```bash
# Best choice: Qwen 2.5 14B Instruct AWQ
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct-awq --eval=healthbench_hard --examples=10

# Alternative: GPTQ version
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct-gptq --eval=healthbench_hard --examples=10
```
**Memory usage**: 4GB (grader) + 7GB (model) = ~11GB total

### Tier 4: 40GB+ GPU (e.g., A100 40GB, L40, A6000)
```bash
# Best quality: Full precision 14B models
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct --eval=healthbench_hard --examples=10

# Or larger quantized models (if available)
```
**Memory usage**: 4GB (grader) + 28GB (model) = ~32GB total

### Memory Optimization Tips

1. **Use pre-quantized AWQ/GPTQ models** - Loads much faster than dynamic quantization
2. **Pre-download models** - Use `huggingface-cli download` to avoid hanging during first run
3. **One model at a time** - Models are loaded/unloaded sequentially
4. **Clear cache** between runs: `torch.cuda.empty_cache()` (automatic in code)
5. **For GPUs < 12GB**: Modify grader in [simple_evals.py:411](simple-evals/simple_evals.py#L411) to use `Qwen/Qwen2.5-3B-Instruct-AWQ` (~2GB) or run on CPU with `device="cpu"`

## Output

Results saved to `/tmp/` as:
- HTML reports: `{eval}_{model}_{timestamp}.html`
- JSON metrics: `{eval}_{model}_{timestamp}.json`