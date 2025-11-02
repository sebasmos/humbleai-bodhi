#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/prod_8gpu_%j.out
#SBATCH --error=logs/prod_8gpu_%j.err
#SBATCH --job-name=prod_8gpu_h200_full
#SBATCH --mem=1500G

# Production job: FULL H200 node (8 GPUs, 141GB each = 1.1TB total)
# For largest models and comprehensive evaluations

echo "========================================="
echo "Production Job - Full H200 Node (8 GPUs)"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated: 8x H200 GPUs (1.1TB total GPU memory)"
echo "========================================="

# Load miniforge module
module load miniforge/24.3.0-0

# Navigate to project directory
cd /orcd/home/002/sebasmos/code/HumbleAILLMs

# Verify GPU allocation
echo ""
echo "GPU Allocation Verification:"
nvidia-smi --list-gpus
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "========================================="

echo ""
echo "Starting full evaluation..."
echo ""

# Run full evaluation with all 8 H200 GPUs
# CHANGE --eval and --examples as needed for your evaluation
python -m simple-evals.simple_evals \
    --model=Llama-3.1-405B-Instruct-FP8 \
    --eval=gpqa \
    --num-gpus 8 \
    --examples=100

echo ""
echo "========================================="
echo "Production run completed at: $(date)"
echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "========================================="
