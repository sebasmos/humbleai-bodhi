# HealthBench Benchmark Results

## Comprehensive Results: Baseline vs v6 (Curious-Humble)

Testing epistemic reasoning through curiosity and humility across HealthBench datasets.

### Results Summary (10 samples each)

| Model | Mode | Consensus | Hard | HealthBench |
|-------|------|-----------|------|-------------|
| **GPT-4o-mini** | Baseline | 76.7% | 0.0% | 26.8% |
| **GPT-4o-mini** | **v6 (Curious-Humble)** | **88.3%** | **11.8%** | **38.6%** |
| | *Improvement* | *+11.6%* | *+11.8%* | *+11.8%* |
| **GPT-4.1** | Baseline | 93.3% | 24.7% | 47.7% |
| **GPT-4.1** | **v6 (Curious-Humble)** | 88.3% | **42.7%** | **52.0%** |
| | *Improvement* | *-5.0%* | *+18.0%* | *+4.3%* |

### Key Findings

1. **GPT-4o-mini benefits significantly from v6** - consistent +11-12% improvement across all datasets
2. **GPT-4.1 shows mixed results** - improves on hard (+18%) and general healthbench (+4.3%), but slight regression on consensus (-5%)
3. **Biggest gains on hard cases** - v6's curiosity-driven questioning helps most on difficult scenarios
4. **v6 helps smaller models more** - GPT-4o-mini sees larger relative improvements than GPT-4.1

---

## Commands to Reproduce

### Baseline Tests

```bash
# GPT-4o-mini Baseline
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_hard --examples=10
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench --examples=10

# GPT-4.1 Baseline
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench_consensus --examples=10
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench_hard --examples=10
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench --examples=10
```

### v6 (Curious-Humble) Tests

```bash
# GPT-4o-mini v6
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass --tta-calibration=6
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_hard --examples=10 --use-tta --tta-two-pass --tta-calibration=6
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench --examples=10 --use-tta --tta-two-pass --tta-calibration=6

# GPT-4.1 v6
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench_consensus --examples=10 --use-tta --tta-two-pass --tta-calibration=6
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench_hard --examples=10 --use-tta --tta-two-pass --tta-calibration=6
python -m simple-evals.simple_evals --model=gpt-4.1 --eval=healthbench --examples=10 --use-tta --tta-two-pass --tta-calibration=6
```

---

## v6 (Curious-Humble) Approach

The v6 approach uses natural language to embed epistemic virtues:

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

---

## Previous TTA Calibration Experiments (Consensus Only)

| Version | Mode | Score | Notes |
|---------|------|-------|-------|
| Baseline | No enhancement | 85.0% | Reference point |
| TTA single-pass | Internal CoT | 80.0% | Worse than baseline |
| TTA two-pass (original) | Simple analysis -> response | **88.3%** | +3.3% improvement |
| **TTA v6 (curious-humble)** | Curiosity + humility prompts | **88.3%** | **Best for GPT-4o-mini** |
| TTA v0 | Simple prompts | 85.0% | Matches baseline |
| TTA v1 | H*/Q* calibration | 81.7% | Worse |
| TTA v2 | Behavioral instructions | 81.7% | Worse |
| TTA v3 | Key insights only | 85.0% | Matches baseline |
| TTA v4 | Ultra-minimal | 85.0% | Matches baseline |
| TTA v5 | HealthBench-focused | 80.0% | Worse |
| TTA v7 | Simple-actionable | 76.7% | Worse - too brief |

### What Doesn't Work

- H* and Q* formula-based calibration (adds confusion)
- Explicit behavioral instructions ("MUST express uncertainty")
- Complex analysis prompts with numeric uncertainty extraction
- Any approach that over-constrains the model's response
- Too-brief prompts (v7) that lose important context

---

## Previous Results: EUDEAS vs Baseline (20 samples)

| Model | Mode | HealthBench Score | Accuracy | EVS |
|-------|------|------------------|----------|-----|
| GPT-4o-mini | **Baseline** | **88.3%** | 91.7% | - |
| GPT-4o-mini | EUDEAS | 80.0% | 83.3% | 0.83 |

EUDEAS with structured PRECISE-U format hurt accuracy by ~8 points.

---

## Conclusions

1. **v6 (Curious-Humble) consistently helps smaller models** - GPT-4o-mini sees +11-12% gains across all datasets
2. **Natural language virtues work better than formulas** - asking "What I'm unsure about" beats H* calculations
3. **Two-pass reasoning helps** - analyze first, respond second improves accuracy
4. **Biggest gains on hard cases** - curiosity-driven questioning helps most when uncertainty is high
5. **Larger models may not need as much guidance** - GPT-4.1 shows mixed results with v6
6. **Keep it simple** - complex calibration formulas add overhead without benefit
7. **10 samples has high variance** - run 20+ for statistical confidence

### Recommended Next Steps

1. Run 20-sample tests to confirm statistical significance
2. Use v6 (curious-humble) for smaller models like GPT-4o-mini
3. Consider adaptive approach: use v6 for hard cases, baseline for easy ones
4. Test with other model families (Claude, Gemini, Llama)
