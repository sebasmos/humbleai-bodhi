#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=s
#SBATCH --output=llama/s_%j.log
#SBATCH --error=llama/s_%j.log

echo "========================================="
echo "Test Job - 1x H200 GPU"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

module load miniforge/24.3.0-0

cd /orcd/home/002/sebasmos/orcd/pool/code/HumbleAILLMs/

echo ""
echo "GPU Information:"
nvidia-smi --list-gpus
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "========================================="

python -m simple-evals.simple_evals \
    --model=gpt-oss-20b \
    --eval=healthbench_consensus \
    --num-gpus 1 \
    --examples=10

echo ""
echo "========================================="
echo "Test completed at: $(date)"
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "========================================="
