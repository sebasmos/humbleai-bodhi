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

## Prerequisites

1. **Generate sample files** using `notebooks/create_multi_seed_samples.ipynb`:
   - Run with `STRATEGY = 'rss'` to create `data/data-5-seeds-200RSS/`
   - Run with `STRATEGY = 'rs'` to create `data/data-5-seeds-200RS/`

2. **Set your API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start

### Run RSS evaluations only:
```bash
./scripts/run_rss_eval.sh
```

### Run RS evaluations only:
```bash
./scripts/run_rs_eval.sh
```

### Run both (full pipeline):
```bash
./scripts/run_all_eval.sh
```

### Run specific seeds:
```bash
# Only seeds 42 and 43
./scripts/run_rss_eval.sh 42 43
./scripts/run_rs_eval.sh 42 43
```

## Output Structure

```
Results/
├── results-5-seeds-200rss/
│   ├── baseline-seed42/
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS.json
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_allresults.json
│   │   └── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS.html
│   ├── baseline-seed{43,44,45,46}/
│   ├── bodhiv0.1.3-seed42/
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi.json
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi_allresults.json
│   │   └── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi.html
│   └── bodhiv0.1.3-seed{43,44,45,46}/
└── results-5-seeds-200rs/
    └── (same structure as RSS)
```

## Verification

After completion, the scripts automatically verify file counts. For manual verification:

```bash
# Check file counts
for dir in Results/results-5-seeds-200*/*/; do
    count=$(ls -1 "$dir" 2>/dev/null | wc -l)
    echo "$dir: $count files"
done

# Check sample counts in allresults files
for f in Results/results-5-seeds-200*/*/*_allresults.json; do
    count=$(python -c "import json; print(len(json.load(open('$f'))['results']))" 2>/dev/null || echo "ERROR")
    echo "$(basename $(dirname $f)): $count samples"
done
```

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/run_rss_eval.sh` | Run RSS (stratified) evaluations |
| `scripts/run_rs_eval.sh` | Run RS (random) evaluations |
| `scripts/run_all_eval.sh` | Run both pipelines sequentially |

## Analysis

After all evaluations complete:

```bash
jupyter notebook notebooks/analyze_multi_seed_results.ipynb
```

Key metrics to compare:
- Mean score across seeds (RSS vs RS)
- Standard deviation across seeds
- BODHI improvement over baseline
- Theme-level performance variance
