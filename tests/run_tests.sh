#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=gpu_test
#SBATCH --output=test_logs/gpu_test_%j.log
#SBATCH --error=test_logs/gpu_test_%j.log

echo "========================================="
echo "Multi-GPU Diagnostic Tests"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# Create log directory
mkdir -p /home/sebasmos/orcd/pool/code/HumbleAILLMs/tests/test_logs

# Load conda
module load miniforge/24.3.0-0

# Navigate to project root
cd /home/sebasmos/orcd/pool/code/HumbleAILLMs

echo ""
echo "Environment Info:"
echo "-----------------"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

echo "GPU Information:"
echo "----------------"
nvidia-smi --list-gpus
echo ""
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "CUDA Environment Variables:"
echo "---------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo ""

echo "========================================="
echo "Running Quick Tests (no 70B model)"
echo "========================================="

# Run quick tests first (small models, fast)
python -m tests.test_multigpu --test quick --num-gpus 2

TEST_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Quick Tests completed with exit code: $TEST_EXIT_CODE"
echo "========================================="

# If quick tests pass, optionally run large model test
if [ "$TEST_EXIT_CODE" -eq 0 ] && [ "${RUN_LARGE_MODEL_TEST:-0}" -eq 1 ]; then
    echo ""
    echo "========================================="
    echo "Running Large Model Test (Llama-3.3-70B)"
    echo "This will take 5-10 minutes..."
    echo "========================================="

    python -m tests.test_multigpu --test large_model --quantize "${QUANTIZE:-}"

    LARGE_TEST_EXIT_CODE=$?
    echo "Large model test exit code: $LARGE_TEST_EXIT_CODE"
fi

echo ""
echo "========================================="
echo "Tests completed at: $(date)"
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "========================================="

exit $TEST_EXIT_CODE
