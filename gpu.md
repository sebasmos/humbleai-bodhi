

## 1. Quick Start

### First Time Setup
```bash
# SSH to Engaging
ssh sebasmos@orcd-login001.mit.edu

# Navigate to project
cd /orcd/home/002/sebasmos/code/HumbleAILLMs

# Create logs directory
mkdir -p logs

# Run first test (5 minutes)
sbatch slurm_scripts/test_1gpu_quick.sh

# Monitor
squeue --me
tail -f logs/test_1gpu_*.out
```

### Progressive Testing
```bash
# 1. Basic test (5 min) - Verify setup
sbatch slurm_scripts/test_1gpu_quick.sh

# 2. Multi-GPU test (15 min) - Test 2 GPUs
sbatch slurm_scripts/test_2gpu_l40s.sh

# 3. Large model test (30 min) - Test 405B model
sbatch slurm_scripts/test_4gpu_h200.sh

# 4. Production run (hours) - Full evaluation
sbatch slurm_scripts/production_4gpu_h200.sh
```

---

## 2. SLURM Scripts

### Testing Scripts (Run These First)

| Script | GPUs | Model | Examples | Time | Memory | Purpose |
|--------|------|-------|----------|------|--------|---------|
| `test_1gpu_quick.sh` | 1 (any) | gpt-neo-1.3b | 5 | ~5 min | ~3GB | Verify basic setup works |
| `test_2gpu_l40s.sh` | 2 L40S | mixtral-8x22b + 4bit | 10 | ~15 min | 96GB | Test multi-GPU functionality |
| `test_4gpu_h200.sh` | 4 H200 | Llama-405B-FP8 | 10 | ~30 min | 564GB | Test large model loading |

### Production Scripts (After Testing Succeeds)

| Script | GPUs | Model | Examples | Time | Memory | Use Case |
|--------|------|-------|----------|------|--------|----------|
| `production_4gpu_h200.sh` | 4 H200 | Llama-405B-FP8 | 50 | 6 hr | 564GB | Full evaluation runs |
| `production_8gpu_h200.sh` | 8 H200 | Llama-405B-FP8 | 100 | 6 hr | 1.1TB | Comprehensive evaluations |
| `interactive_4gpu_h200.sh` | 4 H200 | - | - | 2 hr | 564GB | Interactive development |

**All scripts configured for**: `/orcd/home/002/sebasmos/code/HumbleAILLMs`

---

## 3. Commands

### Basic Usage
```bash
# Single GPU (automatic)
python -m simple-evals.simple_evals \
    --model=gpt-neo-1.3b \
    --eval=mmlu \
    --examples=10
```

### Multi-GPU (Automatic - Uses All Available)
```bash
python -m simple-evals.simple_evals \
    --model=Llama-3.1-405B-Instruct-FP8 \
    --eval=healthbench_hard \
    --examples=50
```

### Multi-GPU (Specify Count)
```bash
python -m simple-evals.simple_evals \
    --model=mixtral-8x22b-instruct \
    --eval=gpqa \
    --num-gpus 4 \
    --examples=100
```

### With Quantization
```bash
python -m simple-evals.simple_evals \
    --model=mixtral-8x22b-instruct \
    --eval=mmlu \
    --num-gpus 2 \
    --quantize 4bit \
    --examples=50
```

### List Available Models
```bash
python -m simple-evals.simple_evals --list-models
```

---

## 4. GPU Memory Requirements

| Model | Parameters | FP16 Memory | 4-bit Memory | Recommended Setup |
|-------|-----------|-------------|--------------|-------------------|
| Llama-405B-FP8 | 405B | ~400GB | ~120GB | 4x H200 (564GB) |
| Mixtral-8x22B | 176B | ~350GB | ~100GB | 3x H200 or 2x H200 + 4bit |
| Qwen2-72B | 72B | ~145GB | ~40GB | 2x H100 or 1x H200 |
| Llama-3-70B | 70B | ~140GB | ~40GB | 1x H200 or 2x L40S |
| Qwen3-32B | 32B | ~65GB | ~20GB | 2x L40S or 1x L40S + 4bit |
| GPT-Neo-1.3B | 1.3B | ~3GB | ~1GB | 1x any GPU |

### GPU Types on Engaging

| GPU Type | Memory | Quantity | Best For |
|----------|--------|----------|----------|
| H200 | 141GB | 104 (8/node) | Large models (405B) |
| L40S | 48GB | 200 (4/node) | Medium models, most available |
| H100 | 80GB | ~32 (4/node) | Training, large models |
| A100 | 40-80GB | ~64 (4/node) | General purpose |

---

## 5. Available Models

### Large Models (Require 2+ GPUs)
- `Llama-3.1-405B-Instruct-FP8` - 405B, FP8 quantized
- `Llama-3.1-405B-FP8` - 405B, FP8 quantized
- `mixtral-8x22b-instruct` - 176B
- `mistral-large-instruct` - 123B

### Medium Models (1-2 GPUs)
- `llama-3.3-70b-instruct` - 70B
- `llama-3-70b` - 70B
- `qwen2-72b-instruct` - 72B
- `qwen2-57b-a14b-instruct` - 57B
- `qwen3-32b` - 32B
- `deepseek-r1-qwen-32b` - 32B

### Small Models (1 GPU)
- `gpt-neo-1.3b` - 1.3B
- `medgemma-4b-it` - 4B medical
- `medgemma-27b-it` - 27B medical

### Quantized Models (Pre-quantized)
- `qwen2.5-3b-instruct-awq` - 3B AWQ
- `qwen2.5-7b-instruct-awq` - 7B AWQ
- `qwen2.5-14b-instruct-awq` - 14B AWQ
- `qwen3-30b-a3b-instruct-fp8` - 30B FP8

---

## 6. Available Evaluations

| Evaluation | Description | Examples |
|------------|-------------|----------|
| `mmlu` | Massive Multitask Language Understanding | ~15K |
| `gpqa` | Graduate-level science questions | 448 |
| `healthbench_hard` | Medical reasoning benchmark | ~1K |
| `math` | Mathematical problem solving | ~5K |
| `drop` | Reading comprehension | ~9.5K |
| `humaneval` | Code generation | 164 |
| `simpleqa` | Simple question answering | ~4K |

---

## 7. Monitoring

### Check Job Status
```bash
squeue --me                        # List your jobs
scontrol show job <job_id>         # Detailed job info
scancel <job_id>                   # Cancel job
```

### Watch Output
```bash
tail -f logs/test_1gpu_*.out       # Watch standard output
tail -f logs/test_1gpu_*.err       # Watch errors
```

### Monitor GPUs (While Job Runs)
```bash
# Get your node name
squeue --me

# SSH to compute node
ssh node3101  # Replace with your node

# Watch GPU utilization
watch -n 1 nvidia-smi

# Exit: Ctrl+C, then 'exit'
```

### Check Available Nodes
```bash
sinfo -o "%20N %10c %10m %25f %10G" | grep gpu
```

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| Job pending too long | Try L40S instead of H200: `test_2gpu_l40s.sh` |
| CUDA out of memory | Add `--quantize 4bit` or increase `--num-gpus` |
| Job fails immediately | Check error log: `cat logs/*_<job_id>.err` |
| GPUs not all used | Verify `--num-gpus` flag is set in command |
| Model not found | List models: `python -m simple-evals.simple_evals --list-models` |
| Module not found | Ensure on compute node, not login node |
| "Already borrowed" error | FP8 dtype fixed - update code if old version |

### Common Error Logs
```bash
# Check error log
cat logs/test_1gpu_<job_id>.err

# Check SLURM account info
sacct -j <job_id> --format=JobID,JobName,State,ExitCode

# Check node allocation
scontrol show job <job_id>
```

---
