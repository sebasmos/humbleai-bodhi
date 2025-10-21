# Simple-Evals Setup Guide

## Installation

```bash
# Clone repository
git clone https://github.com/sebasmos/HumbleAILLMs.git

# Install dependencies
pip install torch transformers accelerate openai anthropic human-eval bitsandbytes

# Optional: Set API keys for cloud models (not needed for local models)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Optional: Set HuggingFace token for gated models (MedGemma, etc.)
export HF_TOKEN="your-hf-token-here"  # Get from https://huggingface.co/settings/tokens
```

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

# Qwen and DeepSeek Models (NEW - High Performance)
# Full precision (FP16) variants
qwen2.5-14b-instruct      # MMLU 79.7%, 14.8B params (~28GB VRAM)
qwen2.5-14b               # Base model (~28GB VRAM)
qwen3-32b                 # MMLU 83-85%, exceptional performance (~64GB VRAM)
deepseek-r1-qwen-32b      # Beats o1-mini, reasoning-focused (~64GB VRAM)
qwen3-30b-a3b             # 3B active params, ArenaHard 91.0 (~60GB VRAM)

# 4-bit quantized variants (RECOMMENDED for GPU memory efficiency)
qwen2.5-14b-instruct-4bit # MMLU 79.7% (~5-7GB VRAM) ⭐ BEST FOR 8-16GB GPU
qwen2.5-14b-4bit          # Base model (~5-7GB VRAM)
qwen3-32b-4bit            # MMLU 83-85% (~14-16GB VRAM) ⭐ BEST FOR 24GB+ GPU
deepseek-r1-qwen-32b-4bit # Reasoning model (~14-16GB VRAM)
qwen3-30b-a3b-4bit        # ArenaHard 91.0 (~13-15GB VRAM)
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

The evaluation system uses **CPU-based graders** (~6GB system RAM) to free GPU memory for the model being evaluated.

### Tier 1: 8-16GB GPU (e.g., RTX 3060, RTX 4060 Ti)
```bash
# Best choice: Qwen 2.5 14B Instruct (4-bit)
python -m simple-evals.simple_evals --model=qwen2.5-14b-instruct-4bit --eval=mmlu

# Alternative: Base model for completion-style tasks
python -m simple-evals.simple_evals --model=qwen2.5-14b-4bit --eval=mmlu
```
**Memory usage**: ~5-7GB GPU + ~6GB system RAM for graders

### Tier 2: 24GB GPU (e.g., RTX 3090, RTX 4090, A5000)
```bash
# Best performance: Qwen 3 32B (4-bit)
python -m simple-evals.simple_evals --model=qwen3-32b-4bit --eval=mmlu

# Reasoning specialist: DeepSeek R1 (4-bit)
python -m simple-evals.simple_evals --model=deepseek-r1-qwen-32b-4bit --eval=math,gpqa

# Fast inference: Qwen 3 30B A3B (4-bit)
python -m simple-evals.simple_evals --model=qwen3-30b-a3b-4bit --eval=mmlu
```
**Memory usage**: ~13-16GB GPU + ~6GB system RAM for graders

### Tier 3: 40GB+ GPU (e.g., A100, L4, L40)
```bash
# Full precision: Qwen 3 32B (best accuracy)
python -m simple-evals.simple_evals --model=qwen3-32b --eval=mmlu

# Full precision: DeepSeek R1 (reasoning tasks)
python -m simple-evals.simple_evals --model=deepseek-r1-qwen-32b --eval=math,gpqa
```
**Memory usage**: ~60-64GB GPU + ~6GB system RAM for graders

### Tier 4: 80GB+ GPU (e.g., A100 80GB, H100)
```bash
# Run multiple models in batch
python -m simple-evals.simple_evals --model=qwen3-32b,deepseek-r1-qwen-32b --eval=mmlu,math
```

### Memory Optimization Tips

1. **Use 4-bit quantization** for 75% memory reduction with minimal accuracy loss
2. **Graders run on CPU** - GPT-neo-1.3B uses system RAM, not GPU
3. **One model at a time** - Models are loaded/unloaded sequentially
4. **Clear cache** between runs: `torch.cuda.empty_cache()` (automatic in code)

## Output

Results saved to `/tmp/` as:
- HTML reports: `{eval}_{model}_{timestamp}.html`
- JSON metrics: `{eval}_{model}_{timestamp}.json`