#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_2gpu_%j.out
#SBATCH --error=logs/test_2gpu_%j.err
#SBATCH --job-name=test_2gpu_l40s

# Test job: 2x L40S GPUs (48GB each = 96GB total)
# Medium model with quantization - 10 examples

echo "========================================="
echo "Test Job - 2x L40S GPUs"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated: 2x L40S GPUs (96GB total)"
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

# Run test with 2 GPUs
python -m simple-evals.simple_evals \
    --model=mixtral-8x22b-instruct \
    --eval=mmlu \
    --num-gpus 2 \
    --quantize 4bit \
    --examples=10

echo ""
echo "========================================="
echo "Test completed at: $(date)"
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "========================================="
