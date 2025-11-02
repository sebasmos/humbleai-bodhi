#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/prod_4gpu_%j.out
#SBATCH --error=logs/prod_4gpu_%j.err
#SBATCH --job-name=prod_4gpu_h200
#SBATCH --mem=500G

# Production job: 4x H200 GPUs (141GB each = 564GB total)
# Full evaluation run with large model

echo "========================================="
echo "Production Job - 4x H200 GPUs"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated: 4x H200 GPUs (564GB total)"
echo "========================================="

# Load miniforge module
module load miniforge/24.3.0-0

# Navigate to project directory
cd /orcd/home/002/sebasmos/code/HumbleAILLMs

# Verify GPU allocation
echo ""
echo "GPU Information:"
nvidia-smi --list-gpus
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "========================================="

echo ""
echo "Starting full evaluation..."
echo ""

# Run full evaluation with 4 H200 GPUs
# CHANGE --eval and --examples as needed for your evaluation
python -m simple-evals.simple_evals \
    --model=Llama-3.1-405B-Instruct-FP8 \
    --eval=healthbench_hard \
    --num-gpus 4 \
    --examples=50

echo ""
echo "========================================="
echo "Production run completed at: $(date)"
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo "========================================="
