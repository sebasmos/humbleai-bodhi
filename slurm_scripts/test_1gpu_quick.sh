#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_1gpu_%j.out
#SBATCH --error=logs/test_1gpu_%j.err
#SBATCH --job-name=test_1gpu

# Quick test job: 1 GPU, small model, 5 examples
# Use this to verify everything works before scaling up

echo "========================================="
echo "Quick Test Job - 1 GPU"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# Load miniforge module
module load miniforge/24.3.0-0

# Navigate to project directory
cd /orcd/home/002/sebasmos/code/HumbleAILLMs

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --list-gpus
echo ""

# Run quick test with small model (5 examples)
python -m simple-evals.simple_evals \
    --model=gpt-neo-1.3b \
    --eval=mmlu \
    --examples=5

echo ""
echo "========================================="
echo "Test completed at: $(date)"
echo "========================================="
