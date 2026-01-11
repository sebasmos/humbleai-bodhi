# HealthBench Consensus Benchmark Results

## EUDEAS vs Baseline Comparison (20 samples)

Testing whether EUDEAS (epistemic reasoning prompts) improves HealthBench accuracy.

### Summary Results

| Model | Mode | HealthBench Score | Accuracy | EVS |
|-------|------|------------------|----------|-----|
| GPT-4o-mini | **Baseline** | **88.3%** | 91.7% | - |
| GPT-4o-mini | EUDEAS | 80.0% | 83.3% | 0.83 |

### Key Findings

1. **Baseline outperforms EUDEAS** by ~8 percentage points
2. EUDEAS provides EVS metrics (epistemic calibration) but at a cost to accuracy
3. The structured PRECISE-U reasoning may be adding overhead without improving responses

### EUDEAS Metrics (20 samples)

| Metric | Value | Description |
|--------|-------|-------------|
| EVS | 0.83 | Epistemic Virtues Score |
| Humility | 0.72 | Expressed uncertainty |
| Curiosity | 0.43 | Information-seeking |
| Confidence | 0.55 | Model confidence |
| Total Uncertainty | 0.29 | Combined uncertainty |

### Implementation Notes

The EUDEAS implementation uses:
1. **PRECISE-U prompting** - Structured epistemic reasoning template
2. **Natural response extraction** - Final response for grader evaluation
3. **EVS calculation** - Calibration metrics from structured output

The model performs PRECISE-U reasoning internally, then generates a natural response that gets evaluated by the HealthBench grader.

### Output Files

- Baseline: `/tmp/healthbench_consensus_gpt-4o-mini_20260111_132143.json`
- EUDEAS: `/tmp/healthbench_consensus_gpt-4o-mini_20260111_133001_eudeas.json`

### Commands

```bash
# Baseline
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=20

# EUDEAS mode
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=20 --use-eudeas
```

### Next Steps for Improvement

To make EUDEAS improve accuracy:
1. Simplify the PRECISE-U template (less overhead)
2. Better integration of epistemic reasoning into the response
3. Test with larger/more capable models (GPT-4o, Claude)
4. Tune the prompt to emphasize accuracy over structure
