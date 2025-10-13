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
gpt-neo-1.3b         # 1.3B parameter general model
gpt-oss-20b          # 20B parameter model (requires >6GB VRAM or CPU)

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

## Output

Results saved to `/tmp/` as:
- HTML reports: `{eval}_{model}_{timestamp}.html`
- JSON metrics: `{eval}_{model}_{timestamp}.json`