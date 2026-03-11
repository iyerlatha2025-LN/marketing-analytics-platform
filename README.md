# 🎯 Marketing Analytics Platform

**Predictive Campaign Optimization · Customer Segmentation · Agentic AI Extensions**

> **Author:** Latha Iyer | M.S. Business Analytics, University of Louisville  
> **Stack:** Python · pandas · scikit-learn · XGBoost · Anthropic Claude API · matplotlib · seaborn

---

## 📓 Notebook

| Notebook | Modules | Description |
|----------|---------|-------------|
| `Marketing_Analytics_Platform_Complete.ipynb` | 00–13 | Full pipeline: data → ML → agentic AI |

## 📦 Modules

### Base (00–09)
| # | Module | Key Output |
|---|--------|-----------|
| 00 | Data Source | Smart loader · iFood 2,205 customers · data dictionary |
| 01 | Setup & Data | Dependencies · dataset validation |
| 01b | Data Cleansing | Missing values · outlier capping · encoding |
| 02 | Feature Engineering | 22 new features: RFM, spend ratios, tenure bins |
| 03 | EDA | Segment analysis · channel distributions |
| 04 | Correlation Analysis | Heatmap · ranked response predictors |
| 05 | Random Forest | AUC 0.88 · feature importance · ROC |
| 06 | XGBoost | AUC 0.91 · model comparison |
| 07 | Recommender System | ALS Collaborative Filtering · Precision@2: 99.9% |
| 08 | Budget Simulator | $50K → $186K expected revenue · 272% ROI |
| 09 | Final Dashboard | 6-panel executive summary |

### Agentic (10–13)
| # | Module | Key Output |
|---|--------|-----------|
| 10 | LLM Insight Narrator | Claude API narrates every chart in plain English |
| 11 | A/B Testing Agent | O'Brien-Fleming sequential boundary · auto LAUNCH / CONTINUE / ABORT |
| 12 | Causal Uplift | T-Learner CATE · Persuadables ~28% · Qini curve · $15–18K recovery |
| 13 | Multi-Agent Pipeline | DataAgent → ModelAgent → ReportAgent → CMO brief |
| 14 | **LITE Framework** | **Latent Intervention Trajectory Estimation — dual-domain causal scoring** |

---

## 🚀 Quick Start

**Google Colab** — upload `ifood_df.csv` via the 📁 sidebar then run all cells.

To activate live Claude narration (Modules 10–13):
1. Get API key at [console.anthropic.com](https://console.anthropic.com)
2. In Colab: click 🔑 **Secrets** → add `ANTHROPIC_API_KEY`
3. Re-run Modules 10–13

**Local:**
```bash
git clone https://github.com/lathaiyer/marketing-analytics-platform
cd marketing-analytics-platform
pip install -r requirements.txt
jupyter notebook
```

---

## 🤖 Agentic Architecture

```
Module 10 — ClaudeNarrator
            reads structured chart data → writes live business commentary
            no human analyst required

Module 11 — ABTestAgent
            design() → monitor() → decide()
            O'Brien-Fleming boundary controls false positives
            autonomous LAUNCH / CONTINUE / ABORT decision

Module 12 — TLearnerUplift
            fit() → predict_cate() → segment() → qini_curve()
            estimates individual causal treatment effect per customer

Module 13 — Multi-Agent Pipeline
            DataAgent → ModelAgent → ReportAgent
            structured context handoffs → CMO executive summary
```

---

## 📊 Results Summary

| Model / System | Metric | Score |
|----------------|--------|-------|
| Random Forest | AUC | 0.88 |
| XGBoost | AUC | 0.91 |
| ALS Recommender | Precision@2 | 99.9% |
| T-Learner Uplift | Persuadables | ~28% of list |
| Budget Optimizer | ROI | 272% ($50K → $186K) |
| Sleeping Dog Recovery | Est. value | $15–18K / quarter |
| LITE MIS | Persuadables identified | ~28% of list |
| LITE BPS | Treatment Responders | ~31% of patients (simulated) |

---


---

## 🔬 LITE Algorithm — Formal Research Note

### Latent Intervention Trajectory Estimation

**LITE** is a domain-agnostic causal scoring algorithm introduced in this platform that
combines three previously separate methodological traditions into a single composite score:

```
LITE(i) = CATE(i)  ×  trajectory_slope(i)  ×  affinity_score(i)

CATE(i)              — T-Learner individual causal effect estimate
                       did the intervention change outcome for individual i?
                       (Künzel et al., 2019 — Metalearners for CATE estimation)

trajectory_slope(i)  — dF/dt from ALS latent factor time series
                       is the individual's hidden profile improving or declining?
                       positive → engaging / responding
                       negative → disengaging / deteriorating ← earliest warning signal

affinity_score(i)    — ALS latent factor magnitude
                       how strongly does individual match the intervention type?
                       (Hu et al., 2008 — Collaborative Filtering via ALS)
```

### Two Output Scores, One Algorithm

| Score | Domain | Positive | Zero | Negative |
|---|---|---|---|---|
| **Marketing Influencer Score (MIS)** | Customer behaviour | Campaign induces purchase | No causal effect | Campaign reduces intent |
| **Biomarker Progression Score (BPS)** | Patient response | Treatment slows progression | No treatment effect | Adverse response |

### The Four Segments (Both Domains)

| LITE Score | Slope | Marketing | Healthcare | Action |
|---|---|---|---|---|
| > +0.10 | any | Persuadable | Treatment Responder | Prioritise |
| ≈ 0, slope ≈ 0 | ≈ 0 | Sure Thing | Stable | Maintain |
| < 0 | any | Sleeping Dog | Adverse Responder | Suppress / Exclude |
| ≈ 0, slope < −0.05 | declining | Lapsing | Fast Progressor | Urgent intervention |

### Key Innovation

Existing uplift models (CausalML, EconML) estimate CATE at a point in time.
Existing ALS recommenders estimate affinity at a point in time.
Neither combines them with a temporal trajectory component.

LITE is the first framework to unify all three into a single score,
making it applicable to any domain where:
- Individual behaviour can be observed across signals (matrix factorisation)
- An intervention is randomly assigned (causal identification)
- Outcomes are measured over time (trajectory estimation)

### Implementation

```python
from src.lite import marketing_lite, healthcare_lite

# Marketing domain
model = marketing_lite(n_factors=8)
model.fit(X, treatment, outcome)
scores = model.score(X)
# scores.columns: cate, trajectory_slope, affinity_score, lite_score, segment, priority_rank

# Healthcare domain — identical API
model = healthcare_lite(n_factors=8)
model.fit(X_patients, treatment, response)
scores = model.score(X_patients)
# Segments: Treatment Responder, Stable, Adverse Responder, Fast Progressor
```

See `src/lite.py` for full implementation and `notebooks/` Module 14 for
worked examples in both domains with visualisations.

## 🧬 Cross-Domain Framework Note

> *The mathematical frameworks in this platform are domain-agnostic. The same models that predict customer purchase behaviour apply directly to patient response and engagement in clinical and neurological research. Both are expressions of the same underlying question: does an intervention change the behaviour of this specific individual, and by how much?*

---

### The Unified Engagement Model

At its core this platform models one fundamental question:

```
Given everything we know about this individual —
does an intervention change their behaviour?
And by how much?
```

That question has two parallel expressions:

| Dimension | Customer Behaviour (Marketing) | Patient Response (Healthcare) |
|---|---|---|
| **The individual** | Customer | Patient |
| **The intervention** | Campaign (email, catalog, offer) | Treatment (drug, therapy, protocol) |
| **The behaviour** | Purchase decision | Treatment response / disease progression |
| **Treated group** | Received the campaign | Received the intervention |
| **Control group** | Never received campaign | Standard care / placebo |
| **CATE / Uplift score** | Did campaign induce the purchase? | Did treatment slow progression? |
| **Persuadable** | Buys only because of campaign | Responds only because of treatment |
| **Sure Thing** | Buys regardless of campaign | Recovers regardless of treatment |
| **Sleeping Dog** | Harmed by contact | Adverse reaction to treatment |
| **Qini curve** | Optimal targeting cutoff | Optimal patient selection for trial |
| **AUUC** | Value of targeting model | Value of patient stratification model |

The mathematics is identical. The domain changes. The insight does not.

---

### ALS — Two Frameworks, One Mathematical Core

This platform's recommender (Module 07) uses **ALS — Alternating Least Squares**, a matrix
factorisation algorithm that discovers latent (hidden) behavioural patterns without being told
what to look for. In an entirely separate domain, **ALS — Amyotrophic Lateral Sclerosis**
research uses structurally identical latent factor models to track neurodegeneration.

**Marketing ALS (Alternating Least Squares)**
```
Observed matrix  →  customer × channel purchase behaviour
Hidden factors   →  latent preferences the customer never stated
                    "premium catalog buyer" emerges from behaviour
                    not from a survey or explicit label

Output           →  personalised channel affinity score per customer
                    routes each customer to their highest-conversion channel
```

**Neurological ALS Research**
```
Observed matrix  →  patient × biomarker measurements over time
Hidden factors   →  latent disease progression trajectory
                    "fast progressor" identified before symptoms peak
                    inferred from mitochondrial and neurological signals

Output           →  personalised progression score per patient
                    stratifies patients for clinical trial eligibility
```

Both use the same alternating optimisation — fix one set of latent factors, solve for
the other, repeat until convergence. The core insight in both cases is that
**the most important signal is hidden**, and must be inferred from observable
behaviour rather than stated preference or visible symptom.

---

### The Mitochondrial Signal Parallel

In ALS neurodegeneration research, **mitochondrial dysfunction** is now recognised as
one of the earliest detectable markers — appearing before motor symptoms become clinically
visible. It is the leading edge of the progression signal.

```
ALS Disease Timeline:
Mitochondrial dysfunction → Motor neuron stress → Visible symptoms → Progression
        ↑
    Earliest signal
    Detectable before clinical presentation
```

In customer behaviour, the equivalent earliest signal is **engagement energy decline**:

```
Customer Disengagement Timeline:
Email open rate drops → Purchase frequency drops → Last purchase → Churn
        ↑
    Earliest signal
    Detectable before the customer actually leaves
```

Both follow the same pattern: **latent degradation precedes observable outcome.**
The causal uplift model in Module 12 is designed to detect this early — identifying
customers whose treatment effect is declining before they become Lost Causes, just as
mitochondrial markers identify patients whose neurological trajectory is deteriorating
before motor function visibly fails.

---

### The Temporal Causal Layer

The T-Learner in Module 12 estimates a static CATE — the causal effect at a point in time.
The natural extension, directly analogous to longitudinal ALS clinical trial design,
is a **temporal causal model** that tracks how the individual treatment effect changes
across time:

```
Static (current):
CATE(individual) = P(outcome | treated) − P(outcome | control)
                   measured at one point in time

Temporal extension:
CATE(individual, t) = P(outcome | treated, t) − P(outcome | control, t)
                      tracks whether the intervention effect
                      is increasing, stable, or degrading
                      for this specific individual over time
```

| Marketing | Clinical |
|---|---|
| Customer feature vector X | Patient biomarker profile X |
| Campaign assignment (0/1) | Treatment assignment (0/1) |
| Purchase probability over time | Motor function score over time |
| CATE(customer, t) | Heterogeneous treatment effect(patient, t) |
| Persuadable segment | Treatment-responsive patient subgroup |
| Sleeping Dog segment | Adverse responder subgroup |
| Qini curve optimal cutoff | Clinical trial optimal enrolment criteria |

---

### Why This Matters

These are not coincidental similarities. Causal inference as a discipline originated in
**randomised controlled trials** in clinical research and was later adopted by technology
and marketing for A/B testing and uplift modelling. The experimental design —
treated group, control group, random assignment, CATE estimation — is identical because
the underlying scientific question is identical:

> *Does this specific intervention change the outcome for this specific individual —*  
> *and by how much?*

Whether the individual is a **customer deciding to purchase** or a **patient responding
to a therapy** — whether the intervention is a marketing campaign or a clinical treatment —
whether the outcome is revenue or motor function — the causal framework is the same.

This platform demonstrates that framework in the marketing domain. The same codebase,
with domain-appropriate data and outcome variables substituted, applies directly to
patient engagement, treatment response stratification, and clinical trial design.

---

*M.S. Business Analytics · University of Louisville · Latha Iyer*  
*github.com/lathaiyer/marketing-analytics-platform*


notepad "C:\Users\latha\marketing-analytics-platform-\README.md"
```

Find the **Overview** section and add these two lines right after it:
```
📄 [Campaign Effectiveness One-Pager](docs/Campaign_Effectiveness_OnePager.pdf)  
📊 [Executive Analytics Briefing](docs/Executive_Analytics_BriefingVF.pptx)