# HumbleAI-HealthBench

[![PyPI](https://img.shields.io/pypi/v/bodhi-llm)](https://pypi.org/project/bodhi-llm/)
[![PLOS Digital Health](https://img.shields.io/badge/PLOS_Digital_Health-10.1371/journal.pdig.0001013-blue)](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001013)
[![The Lancet](https://img.shields.io/badge/The_Lancet-10.1016/S0140--6736(25)01626--5-red)](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(25)01626-5/fulltext)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Evaluation framework for BODHI on HealthBench - measuring epistemic humility and curiosity in medical AI. Supports both API models (OpenAI, Anthropic) and free open-source models from Hugging Face.

**BODHI implementation**: [github.com/sebasmos/bodhi-llm](https://github.com/sebasmos/bodhi-llm) (`pip install bodhi-llm`)

## Quick Start

```bash
# Clone and install
git clone https://github.com/sebasmos/humbleai-healthbench.git
cd humbleai-healthbench
pip install -r requirements.txt

# Install BODHI (latest version)
pip install bodhi-llm[openai]

# Or install a specific version for reproducibility
pip install bodhi-llm[openai]==0.1.2
```

## Reproducibility

To reproduce results from a specific experiment, install the exact BODHI version used:

```bash
# Install specific version from PyPI
pip install bodhi-llm==0.1.2

# Or install from GitHub tag
pip install git+https://github.com/sebasmos/bodhi-llms.git@v0.1.2

# Verify installed version
pip show bodhi-llm | grep Version
```

**Version history**: See [bodhi-llms releases](https://github.com/sebasmos/bodhi-llms/releases) for all versions and changelogs.

## Environment Variables

```bash
# For API model evaluation
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# For gated models (MedGemma, Llama, etc.)
export HF_TOKEN="your-hf-token-here"
```

## Download Models (Optional)

The default grader is `Qwen/Qwen2.5-14B-Instruct-AWQ` (~7GB VRAM). Pre-download for faster runs:

```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ
```

**By GPU tier:**
- **12-16GB**: `Qwen/Qwen2.5-3B-Instruct-AWQ`, `Qwen/Qwen2.5-7B-Instruct-GPTQ`
- **16-24GB**: `Qwen/Qwen2.5-7B-Instruct-AWQ`, `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4`
- **24-48GB**: `Qwen/Qwen2.5-14B-Instruct-AWQ`
- **96GB+**: `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2-72B-Instruct`

## Available Models

```bash
# List all available models
python -m simple-evals.simple_evals --list-models
```

**Free Hugging Face models (no API key needed):**
- Qwen: `qwen2.5-3b-instruct-awq`, `qwen2.5-7b-instruct-awq`, `qwen2.5-14b-instruct-awq`
- Medical: `medgemma-4b-it`, `medgemma-27b-it` (requires free HF_TOKEN)
- Large: `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2-72B-Instruct`

**API models:**
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-3-5-sonnet`

## BODHI Mode

BODHI adds epistemic virtues (curiosity and humility) through two-pass prompting.

```bash
# Run with BODHI
python -m simple-evals.simple_evals \
  --model gpt-4o-mini \
  --eval healthbench_hard \
  --use-bodhi \
  --n-threads 10

# Run on 200 stratified samples
python -m simple-evals.simple_evals \
  --model gpt-4o-mini \
  --eval healthbench_hard \
  --use-bodhi \
  --n-threads 10 \
  --sample-file data/hard_200_sample_ids.json
```

## HealthBench Evaluations

```bash
# Quick test
python -m simple-evals.simple_evals --model gpt-4o-mini --eval healthbench_hard --examples 5

# Local model
python -m simple-evals.simple_evals --model qwen2.5-14b-instruct-awq --eval healthbench_hard --examples 10

# All variants
python -m simple-evals.simple_evals --model gpt-4o-mini --eval healthbench,healthbench_hard,healthbench_consensus --examples 10
```

## Key Parameters

- `--use-bodhi` - Enable BODHI epistemic reasoning
- `--sample-file` - Use specific sample IDs from JSON file
- `--output-dir` - Custom output directory
- `--n-threads` - Parallel threads (default: 1)
- `--examples` - Limit samples per evaluation
- `--quantize 4bit` - Quantize large models

## Output

Results saved to `Results/`:
- `{eval}_{model}_{timestamp}_bodhi.html` - Interactive report
- `{eval}_{model}_{timestamp}_bodhi.json` - Metrics
- `{eval}_{model}_{timestamp}_bodhi_allresults.json` - Full results

---

## Citation

```bibtex
@article{cajas2026beyond,
  title={Beyond overconfidence: Embedding curiosity and humility for ethical medical AI},
  author={Cajas Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s and others},
  journal={PLOS Digital Health},
  volume={5},
  number={1},
  pages={e0001013},
  year={2026}
}

@article{ordonez2025humility,
  title={Humility and curiosity in human--AI systems for health care},
  author={Ordo{\~n}ez, Sebasti{\'a}n Andr{\'e}s Cajas and others},
  journal={The Lancet},
  volume={406},
  number={10505},
  pages={804--805},
  year={2025}
}
```
