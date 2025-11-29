#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=llama
#SBATCH --output=llama/llama_%j.log
#SBATCH --error=llama/llama_%j.log

echo "========================================="
echo "Test Job - 1x H200 GPU"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

module load miniforge/24.3.0-0

pip install sentencepiece

cd  /orcd/home/002/sebasmos/orcd/pool/code/HumbleAILLMs/

echo ""
echo "GPU Information:"
nvidia-smi --list-gpus
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "========================================="

python -m simple-evals.simple_evals \
    --model=gpt-neo-1.3b \
    --eval=healthbench_consensus \
    --num-gpus 2 \
    --examples=10 \
    --n-threads 1 

echo ""
echo "========================================="
echo "Test completed at: $(date)"
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "========================================="
