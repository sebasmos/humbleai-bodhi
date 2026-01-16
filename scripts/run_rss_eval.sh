#!/bin/bash
# =============================================================================
# RSS (Random Stratified Sampling) Evaluation Pipeline
# =============================================================================
# Runs HealthBench Hard evaluations on 5 stratified samples (seeds 42-46)
# for both baseline and BODHI v0.1.3
#
# Usage:
#   export OPENAI_API_KEY="your-api-key"
#   ./scripts/run_rss_eval.sh
#
# Or run specific seeds:
#   ./scripts/run_rss_eval.sh 42 43
# =============================================================================

set -e

# Configuration
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: Please set OPENAI_API_KEY}"
MODEL="gpt-4o-mini"
EVAL="healthbench_hard"
THREADS=10
STRATEGY="rss"
DATA_DIR="data/data-5-seeds-200RSS"
RESULTS_DIR="Results/results-5-seeds-200rss"

# Seeds to run (default: all 5)
if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(42 43 44 45 46)
fi

echo "========================================="
echo "RSS Evaluation Pipeline"
echo "========================================="
echo "Model: ${MODEL}"
echo "Eval: ${EVAL}"
echo "Seeds: ${SEEDS[*]}"
echo "Data: ${DATA_DIR}"
echo "Output: ${RESULTS_DIR}"
echo "Started: $(date)"
echo "========================================="

# Create directories
for seed in "${SEEDS[@]}"; do
    mkdir -p "${RESULTS_DIR}/baseline-seed${seed}"
    mkdir -p "${RESULTS_DIR}/bodhiv0.1.3-seed${seed}"
done

# Track progress
TOTAL=$((${#SEEDS[@]} * 2))
CURRENT=0

# Run Baseline evaluations
echo ""
echo "========================================="
echo "Phase 1: Baseline Evaluations"
echo "========================================="

for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL}] Baseline seed ${seed} - $(date)"
    echo "-----------------------------------------"

    python -m simple-evals.simple_evals \
        --model=${MODEL} \
        --eval=${EVAL} \
        --sample-file="${DATA_DIR}/hard_200_sample_seed${seed}.json" \
        --output-dir="${RESULTS_DIR}/baseline-seed${seed}" \
        --n-threads=${THREADS}

    echo "Completed baseline seed ${seed}"
done

# Run BODHI evaluations
echo ""
echo "========================================="
echo "Phase 2: BODHI v0.1.3 Evaluations"
echo "========================================="

for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[${CURRENT}/${TOTAL}] BODHI seed ${seed} - $(date)"
    echo "-----------------------------------------"

    python -m simple-evals.simple_evals \
        --model=${MODEL} \
        --eval=${EVAL} \
        --use-bodhi \
        --sample-file="${DATA_DIR}/hard_200_sample_seed${seed}.json" \
        --output-dir="${RESULTS_DIR}/bodhiv0.1.3-seed${seed}" \
        --n-threads=${THREADS}

    echo "Completed BODHI seed ${seed}"
done

echo ""
echo "========================================="
echo "RSS Evaluation Pipeline Complete!"
echo "========================================="
echo "Finished: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo ""

# Verification
echo "Verification:"
for seed in "${SEEDS[@]}"; do
    baseline_count=$(ls -1 "${RESULTS_DIR}/baseline-seed${seed}" 2>/dev/null | wc -l | tr -d ' ')
    bodhi_count=$(ls -1 "${RESULTS_DIR}/bodhiv0.1.3-seed${seed}" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Seed ${seed}: baseline=${baseline_count} files, bodhi=${bodhi_count} files"
done
