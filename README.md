# Simple-Evals Setup Guide

## Installation

```bash
# Clone repository
git clone https://github.com/sebasmos/HumbleAILLMs.git

# Install dependencies
pip install torch transformers accelerate openai anthropic human-eval

# Optional: Set API keys for cloud models (not needed for local models)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Quick Test

```bash
# List available models
python -m simple-evals.simple_evals --list-models

# Test with API model
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=mmlu --examples=10

# Test with local model
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=mmlu --examples=10
```

## Basic Evaluations (Local Models)

```bash
# Multiple choice (no grader needed)
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=mmlu --examples=10 --debug
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=gpqa --examples=5 --debug

# Math and reasoning
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=mgsm --examples=5 --debug
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=drop --examples=5 --debug
```

## HealthBench Evaluations

```bash
# Standard HealthBench
python -m simple-evals.simple_evals --model=gpt-neo-1.3b --eval=healthbench --examples=10 --debug

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