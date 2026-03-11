# 🎯 Marketing Analytics Platform

**Predictive Campaign Optimization · Customer Segmentation · Agentic AI Extensions**

> **Author:** Latha Iyer | M.S. Business Analytics, University of Louisville  
> **Stack:** Python · pandas · scikit-learn · XGBoost · Anthropic Claude API · matplotlib · seaborn

---

## 📓 Notebooks

| Notebook | Modules | Description |
|----------|---------|-------------|
| `Marketing_Analytics_Platform.ipynb` | 01–09 | Core ML pipeline |
| `Marketing_Analytics_Platform_Agentic.ipynb` | 10–13 | Agentic AI extensions |

## 📦 Modules

### Base (01–09)
| # | Module | Key Output |
|---|--------|-----------|
| 01 | Setup & Data | 2,240 synthetic customers · 27 features |
| 02 | Feature Engineering | 22 new features: RFM, spend ratios, bins |
| 03 | EDA | Segment analysis, channel distributions |
| 04 | Correlation Analysis | Heatmap + ranked response predictors |
| 05 | Random Forest | AUC ~0.58 · feature importance · ROC |
| 06 | XGBoost | AUC ~0.57 · model comparison |
| 07 | Recommender System | SVD + Content-Based + ALS Hybrid |
| 08 | Budget Simulator | $50K → ~$230K expected revenue |
| 09 | Final Dashboard | 6-panel executive summary |

### Agentic (10–13)
| # | Module | Key Output |
|---|--------|-----------|
| 10 | LLM Insight Narrator | Claude API narrates every chart live |
| 11 | A/B Testing Agent | Power analysis · O'Brien-Fleming · go/no-go |
| 12 | Causal Uplift | T-Learner CATE · Hillstrom segments · Qini |
| 13 | Multi-Agent Pipeline | DataAgent → ModelAgent → ReportAgent |

## 🚀 Quick Start

**Google Colab** — open either notebook in Colab directly.

To activate live Claude narration (Modules 10–13):
1. Get API key at [console.anthropic.com](https://console.anthropic.com)
2. In Colab: click 🔑 **Secrets** → add `ANTHROPIC_API_KEY`
3. Re-run cells

**Local:**
```bash
git clone https://github.com/lathaiyer/marketing-analytics-platform
cd marketing-analytics-platform
pip install -r requirements.txt
jupyter notebook
```

## 🤖 Agentic Architecture

```
Module 10 — ClaudeNarrator: reads chart data → writes live business commentary
Module 11 — ABTestAgent:    design() → monitor() → decide()
Module 12 — TLearnerUplift: fit() → predict_cate() → segment()
Module 13 — Multi-Agent:    DataAgent → ModelAgent → ReportAgent
                                      (structured context handoffs)
```

## 📊 Results Summary

| Model / System | Metric | Score |
|----------------|--------|-------|
| Random Forest | AUC | ~0.58 |
| XGBoost | AUC | ~0.57 |
| ALS Recommender | Precision@2 | ~0.99 |
| T-Learner Uplift | Mean CATE | ~+0.04 |
| Budget Optimizer | ROI | ~361% |

---
*M.S. Business Analytics · University of Louisville · Latha Iyer*
