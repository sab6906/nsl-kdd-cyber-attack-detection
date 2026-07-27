# Predictive Modelling for Cyber Attack Detection (NSL-KDD)

Machine learning-based intrusion detection system comparing four classification algorithms on the NSL-KDD benchmark dataset, with an added cost-based analysis to evaluate real-world deployment trade-offs.

## Overview

Cyber-attacks on enterprises have grown sharply in recent years, and traditional signature-based detection struggles to keep pace with evolving threats. This project builds and evaluates a predictive modelling framework for network intrusion detection, following the CRISP-DM methodology from business understanding through to a cost-based deployment simulation.

## Approach

- **Dataset:** NSL-KDD (an improved, de-duplicated version of KDD'99), ~148,000 network connection records across 41 features
- **Methodology:** CRISP-DM (business understanding → data understanding → preparation → modelling → evaluation → deployment simulation)
- **Preprocessing:** duplicate removal, label standardisation, categorical harmonisation between train/test sets, one-hot encoding, feature scaling
- **Models compared:** Logistic Regression (baseline), Random Forest, XGBoost, Support Vector Machine (linear kernel)
- **Tools:** RStudio — `caret`, `xgboost`, `randomForest`, `e1071`, `ggplot2`, `dplyr`

## Results

| Model | Accuracy | Sensitivity | Specificity | Balanced Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 75.5% | 62.6% | 92.6% | 77.6% | 0.775 |
| Random Forest | 76.3% | 60.4% | 97.3% | 78.9% | 0.962 |
| XGBoost | 78.73% | 64.72% | 97.24% | 80.98% | 0.959 |
| SVM (linear) | 75.3% | 62.3% | 92.4% | 77.4% | 0.795 |

**Cost-based evaluation** (£5,000 per missed attack, £50 per false block):

| Model | Minimum Total Cost |
|---|---|
| Random Forest | **£5.23M** (lowest) |
| SVM | £16.05M |
| Logistic Regression | £16.27M |
| XGBoost | £17.13M |

## Key Findings

- **Ensemble models outperformed linear classifiers** across nearly every metric — consistent with prior IDS research.
- **XGBoost achieved the highest balanced accuracy** (80.98%), making it a strong general-purpose classifier for varied traffic.
- **Random Forest delivered the lowest operational cost and highest specificity**, making it the more economical choice for automated blocking, since it minimises costly false blocks of legitimate traffic.
- **`src_bytes` and `dst_bytes` were the dominant predictive features** in both tree-based models, together accounting for over 70% of XGBoost's predictive gain.
- **Threshold tuning revealed a clear precision/recall trade-off**: pushing the decision threshold to 0.95 raised precision to ~93% but dropped recall to ~55% — illustrating the real operational tension between blocking too aggressively and missing attacks.
- Simulated **real-time auto-blocking** based on prediction probabilities to test how a trained model would behave operationally, not just on static test data.

## Why this matters for SOC/IDS operations

Beyond model accuracy, this project treats model selection as a business decision — weighing the cost of a missed attack against the cost of unnecessarily blocking legitimate traffic. This mirrors how real security operations centres have to balance detection sensitivity against alert fatigue and service disruption, rather than optimising for accuracy alone.

## Repository Contents

- `dissertation.pdf` — full write-up (literature review, methodology, results, discussion)
- `nsl_kdd_analysis.R` — complete R code: cleaning, visualisation, model training, threshold tuning, cost optimisation

## Limitations & Future Work

- NSL-KDD does not reflect modern attack types (APTs, encrypted traffic, IoT-specific attacks)
- Rare attack classes (U2R, R2L) had limited recall without oversampling
- Testing was offline; live-traffic validation (e.g. streaming with Kafka/Spark) is a natural next step
- Explainability methods (SHAP/LIME) would strengthen trust for regulated-industry deployment

---
*Final year analytics project, Edinburgh Napier University.*
