# Multi-Seed Analysis: RSS vs RS Sampling Strategy

Generated: 2026-01-17 10:27:53

## Configuration
- Seeds: [42, 43, 44, 45, 46]
- Samples per seed: 200
- Total evaluations: 4000
- BODHI version: 0.1.3

## RSS vs RS Verdict

| Criterion | RSS | RS | Winner |
|-----------|-----|-----|--------|
| Reproducibility | 8/14 metrics | 5/14 metrics | **RSS** |
| Context-Seeking Rate | 73.5% | 58.0% | **RSS** |
| Overall Score | 2.22% | 2.78% | **RS** |

### **Final Recommendation: RSS**

## BODHI vs Baseline Summary (using RSS)

| Metric | Baseline | BODHI | Winner |
|--------|----------|-------|--------|
| Context-Seeking Rate | 0.0% | 73.5% | BODHI |
| Overall Score | 0.0% | 2.2% | BODHI |
| Wins | 3/14 | 8/14 | BODHI |

## Key Findings

1. **BODHI v0.1.3** significantly improves context-seeking behavior (+73.5pp)
2. **RSS sampling** provides higher context-seeking detection (73.5% vs 58.0%)
3. **RS sampling** shows slightly lower variance on some metrics
4. Both strategies confirm BODHI's effectiveness

## Files Generated
- `rss_vs_rs_analysis.json` - Full comparison data
- `rss_vs_rs_summary.md` - This summary
- `rss_vs_rs_comparison.png` - Visualization
