# Contributing to HumbleAI-HealthBench

Thank you for your interest in contributing to HumbleAI-HealthBench! This guide will help you set up your development environment and ensure your contributions can be properly reviewed and integrated.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reproducing Results](#reproducing-results)

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA-compatible GPU for running evaluations

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/humbleai-healthbench.git
cd humbleai-healthbench
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/sebasmos/humbleai-healthbench.git
```

## Development Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install BODHI Library

```bash
pip install bodhi-llm
```

### 4. Set Up Environment Variables

Create a `.env` file or export directly:

```bash
# For API model evaluation (optional)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# For gated HuggingFace models (optional)
export HF_TOKEN="your-hf-token-here"
```

### 5. Verify Installation

Run the import tests to verify everything is set up correctly:

```bash
python -m tests.test_imports
```

Expected output: All core dependencies should pass.

## Running Tests

### Import Validation Tests

Test that all required packages are installed:

```bash
# Run all import tests
python -m tests.test_imports

# Run with verbose output
python -m tests.test_imports --verbose

# Test specific category
python -m tests.test_imports --category core
python -m tests.test_imports --category ml
python -m tests.test_imports --category gpu
```

Available categories: `stdlib`, `core`, `ml`, `optional`, `human_eval`, `simple_evals`, `gpu`

### EUDEAS Module Tests

Test the EUDEAS (BODHI) implementation:

```bash
python -m tests.test_eudeas
```

### Quick Evaluation Test

Run a quick test to verify the evaluation pipeline works:

```bash
python -m simple-evals.simple_evals \
  --model gpt-4o-mini \
  --eval healthbench_hard \
  --examples 3
```

### Multi-GPU Tests (HPC/Cluster)

For SLURM-based clusters with GPUs:

```bash
sbatch tests/run_tests.sh
```

## Code Style

### Python

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Keep functions focused and documented
- Use descriptive variable names

### Commits

- Write clear, concise commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when applicable (`Fixes #123`)

### Example commit message:

```
Add stratified sampling validation for theme distribution

- Verify theme proportions match original dataset
- Add statistical tests for distribution comparison
- Update documentation with sampling methodology
```

## Submitting Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Run Tests Before Committing

```bash
python -m tests.test_imports
python -m tests.test_eudeas
```

### 4. Commit and Push

```bash
git add .
git commit -m "Your descriptive commit message"
git push origin feature/your-feature-name
```

### 5. Open a Pull Request

- Go to GitHub and create a PR against the `main` branch
- Fill out the PR template with:
  - Description of changes
  - Related issues
  - Testing performed
  - Screenshots (if applicable)

## Reproducing Results

### Dataset

The evaluation uses the HealthBench Hard dataset:

```bash
# Dataset is automatically downloaded and cached
# URL: https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl
```

### Stratified Sampling

To generate reproducible samples:

```bash
# Open the sampling notebook
jupyter notebook notebooks/create_multi_seed_samples.ipynb
```

Configuration options in the notebook:
- `NUM_SEEDS`: Number of random seeds (default: 5)
- `SAMPLE_SIZE`: Samples per seed (default: 200)
- `STRATEGY`: `rss` (stratified) or `rs` (random)
- `BASE_SEED`: Starting seed (default: 42)

### Running Evaluations

#### With BODHI (Recommended)

```bash
python -m simple-evals.simple_evals \
  --model gpt-4o-mini \
  --eval healthbench_hard \
  --use-bodhi \
  --n-threads 10 \
  --sample-file data/data-5-seeds-200RSS/hard_200_sample_seed42.json
```

#### Without BODHI (Baseline)

```bash
python -m simple-evals.simple_evals \
  --model gpt-4o-mini \
  --eval healthbench_hard \
  --n-threads 10 \
  --sample-file data/data-5-seeds-200RSS/hard_200_sample_seed42.json
```

### Local Models

For running with local HuggingFace models:

```bash
# List available models
python -m simple-evals.simple_evals --list-models

# Run with a local model
python -m simple-evals.simple_evals \
  --model qwen2.5-14b-instruct-awq \
  --eval healthbench_hard \
  --use-bodhi \
  --examples 10
```

### Output Structure

Results are saved to `Results/`:
- `{eval}_{model}_{timestamp}.html` - Interactive report
- `{eval}_{model}_{timestamp}.json` - Metrics summary
- `{eval}_{model}_{timestamp}_allresults.json` - Full results with all responses

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- For general questions, use GitHub Discussions

## License

By contributing, you agree that your contributions will be licensed under the CC BY-NC-SA 4.0 license.
