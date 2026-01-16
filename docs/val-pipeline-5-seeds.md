# Validation Pipeline: 5-Seed Evaluation (RSS & RS)

This document describes the validation pipeline for running multi-seed evaluations on HealthBench Hard using both Random Stratified Sampling (RSS) and Random Sampling (RS) strategies.

## Overview

Run HealthBench Hard evaluations on 5 different samples (seeds 42-46) for both baseline and BODHI v0.1.3, comparing results across:
- **Sampling strategies**: RSS (stratified) vs RS (random)
- **Evaluation modes**: Baseline vs BODHI v0.1.3

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `gpt-4o-mini` |
| Evaluation | `healthbench_hard` |
| Seeds | 42, 43, 44, 45, 46 |
| Samples per seed | 200 |
| BODHI version | v0.1.3 |
| Threads | 10 |

## Sample Files

Generated using `notebooks/create_multi_seed_samples.ipynb`:

```
data/
├── data-5-seeds-200RSS/          # Random Stratified Sampling
│   ├── hard_200_sample_seed42.json
│   ├── hard_200_sample_seed43.json
│   ├── hard_200_sample_seed44.json
│   ├── hard_200_sample_seed45.json
│   ├── hard_200_sample_seed46.json
│   └── metadata.json
└── data-5-seeds-200RS/           # Random Sampling
    ├── hard_200_sample_seed42.json
    ├── hard_200_sample_seed43.json
    ├── hard_200_sample_seed44.json
    ├── hard_200_sample_seed45.json
    ├── hard_200_sample_seed46.json
    └── metadata.json
```

## Output Structure

```
Results/
├── results-5-seeds-200rss/
│   ├── baseline-seed42/
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS.json
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_allresults.json
│   │   └── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS.html
│   ├── baseline-seed43/
│   ├── baseline-seed44/
│   ├── baseline-seed45/
│   ├── baseline-seed46/
│   ├── bodhiv0.1.3-seed42/
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi.json
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi_allresults.json
│   │   └── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi.html
│   ├── bodhiv0.1.3-seed43/
│   ├── bodhiv0.1.3-seed44/
│   ├── bodhiv0.1.3-seed45/
│   └── bodhiv0.1.3-seed46/
└── results-5-seeds-200rs/
    ├── baseline-seed42/
    ├── ... (same structure as RSS)
    └── bodhiv0.1.3-seed46/
```

## Implementation Steps

### Step 1: Generate Sample Files

Run the sampling notebook with each strategy:

```python
# In notebooks/create_multi_seed_samples.ipynb
NUM_SEEDS = 5
SAMPLE_SIZE = 200
BASE_SEED = 42

# Run once with:
STRATEGY = 'rss'  # Creates data/data-5-seeds-200RSS/

# Run again with:
STRATEGY = 'rs'   # Creates data/data-5-seeds-200RS/
```

### Step 2: Create Directory Structure

```bash
# RSS results
mkdir -p Results/results-5-seeds-200rss/baseline-seed{42,43,44,45,46}
mkdir -p Results/results-5-seeds-200rss/bodhiv0.1.3-seed{42,43,44,45,46}

# RS results
mkdir -p Results/results-5-seeds-200rs/baseline-seed{42,43,44,45,46}
mkdir -p Results/results-5-seeds-200rs/bodhiv0.1.3-seed{42,43,44,45,46}
```

### Step 3: Run Baseline Evaluations

```bash
export OPENAI_API_KEY="your-api-key-here"

# RSS Baseline
for seed in 42 43 44 45 46; do
  python -m simple-evals.simple_evals \
    --model=gpt-4o-mini \
    --eval=healthbench_hard \
    --sample-file=data/data-5-seeds-200RSS/hard_200_sample_seed${seed}.json \
    --output-dir=Results/results-5-seeds-200rss/baseline-seed${seed} \
    --n-threads=10
done

# RS Baseline
for seed in 42 43 44 45 46; do
  python -m simple-evals.simple_evals \
    --model=gpt-4o-mini \
    --eval=healthbench_hard \
    --sample-file=data/data-5-seeds-200RS/hard_200_sample_seed${seed}.json \
    --output-dir=Results/results-5-seeds-200rs/baseline-seed${seed} \
    --n-threads=10
done
```

### Step 4: Run BODHI v0.1.3 Evaluations

```bash
# RSS BODHI
for seed in 42 43 44 45 46; do
  python -m simple-evals.simple_evals \
    --model=gpt-4o-mini \
    --eval=healthbench_hard \
    --use-bodhi \
    --sample-file=data/data-5-seeds-200RSS/hard_200_sample_seed${seed}.json \
    --output-dir=Results/results-5-seeds-200rss/bodhiv0.1.3-seed${seed} \
    --n-threads=10
done

# RS BODHI
for seed in 42 43 44 45 46; do
  python -m simple-evals.simple_evals \
    --model=gpt-4o-mini \
    --eval=healthbench_hard \
    --use-bodhi \
    --sample-file=data/data-5-seeds-200RS/hard_200_sample_seed${seed}.json \
    --output-dir=Results/results-5-seeds-200rs/bodhiv0.1.3-seed${seed} \
    --n-threads=10
done
```

## Automation Script

Create `scripts/run_multi_seed_eval.sh`:

```bash
#!/bin/bash
set -e

# Configuration
export OPENAI_API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"
MODEL="gpt-4o-mini"
EVAL="healthbench_hard"
THREADS=10
SEEDS=(42 43 44 45 46)

# Run evaluations for a given strategy
run_strategy() {
    local strategy=$1  # rss or rs
    local data_dir="data/data-5-seeds-200${strategy^^}"
    local results_dir="Results/results-5-seeds-200${strategy}"

    echo "========================================="
    echo "Running ${strategy^^} evaluations"
    echo "========================================="

    # Create directories
    for seed in "${SEEDS[@]}"; do
        mkdir -p "${results_dir}/baseline-seed${seed}"
        mkdir -p "${results_dir}/bodhiv0.1.3-seed${seed}"
    done

    # Baseline evaluations
    for seed in "${SEEDS[@]}"; do
        echo "[$(date)] Running baseline seed ${seed} (${strategy^^})..."
        python -m simple-evals.simple_evals \
            --model=${MODEL} \
            --eval=${EVAL} \
            --sample-file="${data_dir}/hard_200_sample_seed${seed}.json" \
            --output-dir="${results_dir}/baseline-seed${seed}" \
            --n-threads=${THREADS}
    done

    # BODHI evaluations
    for seed in "${SEEDS[@]}"; do
        echo "[$(date)] Running BODHI seed ${seed} (${strategy^^})..."
        python -m simple-evals.simple_evals \
            --model=${MODEL} \
            --eval=${EVAL} \
            --use-bodhi \
            --sample-file="${data_dir}/hard_200_sample_seed${seed}.json" \
            --output-dir="${results_dir}/bodhiv0.1.3-seed${seed}" \
            --n-threads=${THREADS}
    done
}

# Run both strategies
run_strategy "rss"
run_strategy "rs"

echo "========================================="
echo "All evaluations complete!"
echo "========================================="
```

## Time Estimates

| Phase | Evaluations | Time per eval | Total |
|-------|-------------|---------------|-------|
| RSS Baseline | 5 | ~15 min | ~1.25 hrs |
| RSS BODHI | 5 | ~20 min | ~1.67 hrs |
| RS Baseline | 5 | ~15 min | ~1.25 hrs |
| RS BODHI | 5 | ~20 min | ~1.67 hrs |
| **Total** | **20** | - | **~6 hrs** |

## Verification Checklist

After completion, verify:

- [ ] Each of the 20 folders contains 3 files (`.json`, `_allresults.json`, `.html`)
- [ ] Each `_allresults.json` contains 200 examples
- [ ] No error messages in evaluation logs
- [ ] Compare scores across seeds to estimate variance

### Quick verification script:

```bash
# Check file counts
for dir in Results/results-5-seeds-200*/*/; do
    count=$(ls -1 "$dir" 2>/dev/null | wc -l)
    echo "$dir: $count files"
done

# Check sample counts in allresults files
for f in Results/results-5-seeds-200*/*/*_allresults.json; do
    count=$(python -c "import json; print(len(json.load(open('$f'))['results']))" 2>/dev/null || echo "ERROR")
    echo "$f: $count samples"
done
```

## Analysis

After all evaluations complete, analyze results using:

```bash
# Open analysis notebook
jupyter notebook notebooks/analyze_multi_seed_results.ipynb
```

Key metrics to compare:
- Mean score across seeds (RSS vs RS)
- Standard deviation across seeds
- BODHI improvement over baseline
- Theme-level performance variance

## Progress Tracking

### RSS Evaluations

| Seed | Baseline | BODHI v0.1.3 |
|------|----------|--------------|
| 42 | [ ] | [ ] |
| 43 | [ ] | [ ] |
| 44 | [x] | [x] |
| 45 | [ ] | [ ] |
| 46 | [ ] | [ ] |

### RS Evaluations

| Seed | Baseline | BODHI v0.1.3 |
|------|----------|--------------|
| 42 | [ ] | [ ] |
| 43 | [ ] | [ ] |
| 44 | [ ] | [ ] |
| 45 | [ ] | [ ] |
| 46 | [ ] | [ ] |
