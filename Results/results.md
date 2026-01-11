# HealthBench Consensus Benchmark Results

## Summary: Two-Pass Think-Then-Answer Experiments

Testing epistemic reasoning approaches to improve HealthBench scores through curiosity and humility.

### Results (10 samples each)

| Version | Mode | Score | Notes |
|---------|------|-------|-------|
| Baseline | No enhancement | 85.0% | Reference point |
| TTA single-pass | Internal CoT | 80.0% | Worse than baseline |
| TTA two-pass (original) | Simple analysis → response | **88.3%** | +3.3% improvement |
| **TTA v6 (curious-humble)** | Curiosity + humility prompts | **88.3%** | **Ties best! Embeds virtues naturally** |
| TTA v0 | Simple prompts | 85.0% | Matches baseline |
| TTA v1 | H*/Q* calibration | 81.7% | Worse |
| TTA v2 | Behavioral instructions | 81.7% | Worse |
| TTA v3 | Key insights only | 85.0% | Matches baseline |
| TTA v4 | Ultra-minimal | 85.0% | Matches baseline |
| TTA v5 | HealthBench-focused | 80.0% | Worse |
| TTA v7 | Simple-actionable | 76.7% | Worse - too brief |

### Key Findings

1. **v6 (curious-humble) ties best at 88.3%** - successfully embeds curiosity and humility
2. **Calibration formulas (H*/Q*) hurt performance** - the math adds overhead without improving responses
3. **Natural language virtues work better than formulas** - asking "What I'm unsure about" beats H* calculations
4. **10 samples has high variance** - need 20+ samples for statistical confidence

### What Works: v6 (Curious-Humble) Approach

The v6 approach that achieved 88.3% uses natural language to embed epistemic virtues:

**Pass 1 Analysis Prompt:**
- "WHAT I THINK: Your best assessment (be honest about confidence)"
- "WHAT I'M UNSURE ABOUT: Key uncertainties that affect your assessment"
- "WHAT I NEED TO KNOW: Questions that would significantly help"
- "RED FLAGS: Any urgent warning signs"
- "SAFE ADVICE: What can you confidently recommend regardless of uncertainty?"

**Pass 2 Response Prompt:**
- Extracts uncertainties and questions from analysis
- Guides model to "ASK them naturally" and "express uncertainty honestly"
- Focuses on being "genuinely helpful, curious, and appropriately humble"

### What Doesn't Work

- H* and Q* formula-based calibration (adds confusion)
- Explicit behavioral instructions ("MUST express uncertainty")
- Complex analysis prompts with numeric uncertainty extraction
- Any approach that over-constrains the model's response
- Too-brief prompts (v7) that lose important context

### Commands

```bash
# Baseline
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10

# Best performing: v6 (curious-humble)
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass --tta-calibration=6

# Original two-pass (also 88.3%)
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass

# Other versions
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass --tta-calibration=0  # simple
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass --tta-calibration=1  # H*/Q*
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass --tta-calibration=7  # simple-actionable
```

---

## Previous Results: EUDEAS vs Baseline (20 samples)

| Model | Mode | HealthBench Score | Accuracy | EVS |
|-------|------|------------------|----------|-----|
| GPT-4o-mini | **Baseline** | **88.3%** | 91.7% | - |
| GPT-4o-mini | EUDEAS | 80.0% | 83.3% | 0.83 |

EUDEAS with structured PRECISE-U format hurt accuracy by ~8 points.

---

## Conclusions

1. **Two-pass reasoning CAN help** - analyze first, respond second (+3.3% over baseline)
2. **Natural curiosity and humility work** - v6 prompts asking "What I'm unsure about" and "What I need to know" achieve best results
3. **Keep it simple** - complex calibration formulas add overhead without benefit
4. **Avoid over-constraining** - behavioral instructions like "MUST express uncertainty" hurt natural responses
5. **H*/Q* formulas don't translate to better responses** - they're more useful for monitoring than steering
6. **Need more samples** - 10 samples has too much variance; run 20+ for reliable comparisons

### Recommended Next Steps

1. Run 20-sample tests with v6 to confirm statistical significance
2. v6 (curious-humble) is the recommended approach for embedding epistemic virtues
3. Consider using H*/Q* for monitoring/logging only, not for response steering
4. Test with larger models (GPT-4o) where reasoning may help more
