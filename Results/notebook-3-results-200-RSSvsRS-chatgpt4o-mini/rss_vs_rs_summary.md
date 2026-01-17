# Notebook 3: RSS vs RS Sampling Strategy Comparison

Generated: 2026-01-17 10:53:05

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

## Key Findings

1. **RSS sampling** provides higher context-seeking detection (73.5% vs 58.0%)
2. **RSS** wins on reproducibility (8/14 metrics with lower variance)
3. **RS** shows slightly higher overall score (2.78% vs 2.22%)
4. Both strategies confirm BODHI's effectiveness over baseline

## Files Generated
- `rss_vs_rs_analysis.json` - Full comparison data
- `rss_vs_rs_summary.md` - This summary
- `rss_vs_rs_comparison.png` - Visualization
