#!/bin/bash
# =============================================================================
# Full Evaluation Pipeline (RSS + RS)
# =============================================================================
# Runs both RSS and RS evaluation pipelines sequentially
#
# Usage:
#   export OPENAI_API_KEY="your-api-key"
#   ./scripts/run_all_eval.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "Full Evaluation Pipeline (RSS + RS)"
echo "========================================="
echo "Started: $(date)"
echo "========================================="

# Run RSS evaluations
echo ""
echo ">>> Running RSS (Random Stratified Sampling) evaluations..."
"${SCRIPT_DIR}/run_rss_eval.sh"

# Run RS evaluations
echo ""
echo ">>> Running RS (Random Sampling) evaluations..."
"${SCRIPT_DIR}/run_rs_eval.sh"

echo ""
echo "========================================="
echo "Full Pipeline Complete!"
echo "========================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  - Results/results-5-seeds-200rss/"
echo "  - Results/results-5-seeds-200rs/"
