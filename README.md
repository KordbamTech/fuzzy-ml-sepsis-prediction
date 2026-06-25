# Hybrid Fuzzy–ML Septic Shock Risk Prediction System

> MSc Artificial Intelligence | ELEC73149 | University of Staffordshire | May 2026

Early prediction of septic shock from continuous ICU physiological data — combining a **Mamdani Fuzzy Inference System** with **LightGBM** in a weighted parallel fusion architecture, with full **SHAP explainability**.

---

## The Problem

Septic shock contributes to over 11 million deaths annually. Conventional ICU alarm systems use rigid thresholds and frequently miss the gradual, multi-attribute physiological deterioration that precedes haemodynamic collapse. This project builds an interpretable hybrid AI system that detects early warning patterns before they become critical.

---

## System Architecture

```
7 Vital-Sign Inputs [HR, SBP, DBP, SpO2, RR, Temp, Stress]
         |
    .----+----.
    |         |
  Mamdani   LightGBM
   FIS        Classifier
  (9 rules)  (200 trees)
    |         |
  F_norm    P_ML
    |         |
    '----+----'
         |
   Weighted Fusion
   H = 0.4 * F_norm + 0.6 * P_ML
         |
   Decision (theta = 0.35)
         |
   ALERT + Rule Trace + SHAP
```

**Why hybrid?** Fuzzy logic captures linguistic clinical concepts ("hypotensive", "tachycardic") and detects pre-shock states where blood pressure is still normal but autonomic stress is elevated — cases the ML model structurally cannot detect from the proxy label. LightGBM provides statistical discriminative power from 144,000 observations. SHAP explains both.

---

## Dataset

- **144,000 rows** — 20 ICU patients monitored at 1-second resolution for 7,200 seconds each
- **16 features** covering vital signs, demographics and physiological signals
- **No pre-existing labels** — supervised proxy target engineered using Sepsis-3 criteria
- **Extreme imbalance**: 60 positive / 143,940 negative (0.042% positive rate)
- Missing values: 7,200 per vital-sign column (single patient sensor dropout) repaired using per-patient linear interpolation

---

## Feature Selection

7 features selected across 3 criteria: Sepsis-3 clinical relevance, data restorability, and modelling coherence:

| Feature | Domain | Clinical Basis |
|---|---|---|
| `heart_rate` | Cardiovascular | Tachycardia >100 bpm: Sepsis-3 criterion |
| `systolic_bp` | Cardiovascular | Hypotension <90 mmHg: primary shock criterion |
| `diastolic_bp` | Cardiovascular | Vascular resistance assessment |
| `spo2` | Respiratory | SpO2 <94%: significant hypoxia |
| `resp_rate` | Respiratory | Tachypnoea >20 bpm: Sepsis-3 criterion |
| `temp_c` | Thermoregulation | Fever >38C or hypothermia <36C |
| `stress_index` | Autonomic | Autonomic dysregulation index |

---

## Fuzzy Inference System

- **Type:** Mamdani (chosen over TSK for linguistic interpretability required in ICU context)
- **Inputs:** 7 vital signs with overlapping trapezoidal and Gaussian membership functions
- **Rules:** 9 clinically grounded rules (capped at 9 to maintain genuine interpretability)
- **Defuzzification:** Centroid, normalised to F_norm in [0,1]
- **Tool:** MATLAB Fuzzy Logic Designer

Key rules:
- `R1`: SBP Low OR (HR High AND SpO2 Low) -> **High risk** (primary shock criteria)
- `R3`: SBP Normal AND HR High AND Stress High -> **Moderate** (pre-shock: BP maintained but autonomic stress elevated)
- `R7`: HR, SBP, SpO2, Temp all Normal -> **Low risk**

---

## Machine Learning Results

| Model | Precision | Recall | F1 | AUC | Notes |
|---|---|---|---|---|---|
| Logistic Regression | 0.197 | 1.000 | 0.329 | 1.000 | 49 false positives |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | Label leakage — excluded |
| LightGBM | 0.414 | 1.000 | 0.585 | 0.9997 | Best credible standalone |
| **Hybrid (a=0.40, t=0.35)** | **0.500** | **1.000** | **0.667** | **0.9999** | **Best overall** |

5-fold CV: Hybrid F1 = 0.666 +/- 0.018 vs LightGBM 0.584 +/- 0.021

---

## Fusion Formula

```
H = 0.40 * F_norm + 0.60 * P_ML

Worked example (HR=112, SBP=87, SpO2=91, RR=24, Temp=38.7, Stress=71):
  FIS:      F(x) = 84.3  ->  F_norm = 0.843
  LightGBM: P_ML  = 0.924
  Hybrid:   H = 0.40x0.843 + 0.60x0.924 = 0.891 >= 0.35  ->  ALERT
  Output:   "Rules R1 and R9 fired. SBP critically low, HR high, SpO2 low.
             SHAP: systolic_bp +0.39, resp_rate +0.11. Vasopressor assessment recommended."
```

Alpha = 0.40 selected by empirical calibration sweep. F1 peaks at alpha=0.40 and falls on both sides, confirming complementary (not redundant) FIS contribution.

---

## Key Results

- **29% reduction in false positives** vs standalone LightGBM (17 -> 12 FP), zero false negatives
- **Dual explanation channel**: SHAP for statistical confidence + FIS rule trace for clinical reasoning
- **Robustness**: Recall = 1.000 maintained under Gaussian noise, 15% missing values, contradictory inputs, alpha-sensitivity sweep
- **Inference time**: ~1.1 ms total — well within 1,000 ms ICU one-second monitoring budget

---

## Tech Stack

```
Python     pandas, numpy, scikit-learn, lightgbm, shap, matplotlib, seaborn
MATLAB     Fuzzy Logic Designer
Tools      Google Colab, Excel
Standards  Sepsis-3 (Singer et al., JAMA 2016)
```

---

## Limitations

- Proxy label != confirmed clinical diagnosis (no vasopressor/lactate data available)
- 20-patient cohort: 95% CI on Recall approx [0.735, 1.000] — too wide for clinical certification
- Patient-stratified and external validation not yet performed

---

## Academic Context

**Module:** ELEC73149 – Artificial Intelligence
**Institution:** University of Staffordshire
**Tutor:** Masum Billah
**Author:** Oluwakorede Solomon Bamidele (ID: 25023666)
