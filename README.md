# Fraud-Radar  

**Project Overview:**  
Fraud-Radar is a **production-style card-fraud detection system**. It turns raw transactional data into **real-time fraud risk scores**, exposes a **FastAPI** microservice with **single & batch** endpoints, and supports **cost-aware decisions** and **explanations**. The stack includes **MySQL**, **LightGBM/XGBoost**, **probability calibration**, and **SHAP**. Secrets are managed via **`.env`**—no passwords in code or requests.

---

## Table of Contents
- [1. Data & Feature Engineering](#1-data--feature-engineering)
- [2. Model Development](#2-model-development)
- [3. Cost-Aware Thresholding](#3-costaware-thresholding)
- [4. API Service (FastAPI)](#4-api-service-fastapi)
- [5. Local Setup & Runbook](#5-local-setup--runbook)
- [6. How to Use the API](#6-how-to-use-the-api)
- [7. Key Results & Metrics](#7-key-results--metrics)
- [8. Model Explainability (SHAP)](#8-model-explainability-shap)
- [9. Versioning & Reproducibility](#9-versioning--reproducibility)
- [10. Security & Secrets](#10-security--secrets)
- [11. Project Structure](#11-project-structure)
- [12. Future Improvements](#12-future-improvements)
- [Built With](#built-with)
- [License](#license)

---

## 1. Data & Feature Engineering

**Source:** MySQL 8 (local). The dataset contains ~**20,000** transactions with a fraud rate around **~5–6%** (highly imbalanced).  

**Feature families (examples):**
- **Transaction**: `amount`, `tx_time`, `night_time`, `country_mismatch`
- **Card velocity**: `card_tx_24h`, `card_amt_24h`, `card_avg_24h`, `card_avg_amt_30d`, `card_min_since_prev`, `card_kmh_from_prev`, `impossible_speed`
- **Device signals**: `device_tx_24h`, `device_tx_30d`, `device_card_count_7d`, `device_age_days`, `device_prior_fraud_30d`, `device_prior_fraud_rate_30d`
- **IP signals**: `ip_tx_30d`, `ip_prior_fraud_30d`, `ip_prior_fraud_rate_30d`, `ip_card_count_7d`
- **Merchant risk**: `merch_chargeback_rate_30d`, `merchant_risk`, `merchant_category`
- **Account**: `account_age_days`
- **Categoricals**: `merchant_category`, `device_os` (one-hot at training/serving)

**At serving time**, the API:
- Pulls the exact feature set via SQL (see `SELECT_COLUMNS` in `src/api/app.py`)
- Applies the same preprocessing:
  - `high_amount_flag`, `log_amount`
  - `amt_over_card_avg30` with safe division
  - Numeric coercion & NaN handling
  - One-hot encoding + **reindex to `feature_columns.json`** so column order matches training

---

## 2. Model Development

Two learners were benchmarked:

- **LightGBM (winner)** — Tuned for imbalanced data, best **Recall @ 1% FPR**  
- **XGBoost (baseline)** — Close in performance, used to validate approach

**Probability calibration**  
- Post-training calibration (isotonic/Platt) to improve probability quality  
- Exported bundle: **`artifacts/model_v12.pkl`** includes **calibrated model** + **base model** (for SHAP)

**Training scripts** (examples):
```bash
# LightGBM (v11)
python src/models/train_lgbm_v11.py

# XGBoost baseline (v12)
python src/models/train_xgb_v12.py
```
## 3. Cost-Aware Thresholding
We optimize the decision threshold to balance false positives (review cost) vs false negatives (fraud loss), with an FPR guardrail (~≤1%).
```bash
python src/models/pick_threshold.py
# outputs best thresholds and cost deltas; default served via THRESHOLD in .env
```
A typical selected threshold in this repo: 0.07093

## 4. API Service (FastAPI)
Why FastAPI?

- Async, OpenAPI docs, easy to deploy.

- Two scoring modes: single tx (/score_tx_id) and batch (/score_many).

- Optional explainability via SHAP.

Endpoints

- GET /health — liveness

- POST /score_tx_id — JSON: { "tx_id": <int>, "threshold": <float?>, "explain": <bool> }

- POST /score_many — JSON: { "tx_ids": [<int>...], "threshold": <float?>, "explain": <bool> }

- GET /metrics — last validation snapshot (val_scores.csv)

## 5. Local Setup & Runbook
5.1 Requirements
```bash
python -V  # 3.10+
pip install fastapi uvicorn joblib numpy pandas scikit-learn lightgbm xgboost shap \
            python-dotenv mysql-connector-python
```
5.2 MySQL (minimal)
```bash
CREATE DATABASE IF NOT EXISTS fraud_radar;

-- Training/serving tables must exist (e.g., train_data_v11 with SELECT_COLUMNS)

CREATE USER IF NOT EXISTS 'fraud_ro'@'localhost' IDENTIFIED BY 'YOUR_STRONG_PASSWORD';
GRANT SELECT ON fraud_radar.* TO 'fraud_ro'@'localhost';
FLUSH PRIVILEGES;
```
5.3 Environment
```bash
cp .env.example .env
```
.env
```bash
DB_HOST=127.0.0.1
DB_NAME=fraud_radar
DB_USER=fraud_ro
DB_PASS=PASSWORD
THRESHOLD=0.07093
```
Note: .env is git-ignored. Only commit .env.example.

5.4 Run the API
```bash
uvicorn src.api.app:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```
## 6. How to Use the API
Single transaction (with explanations)
```bash
curl -X POST "http://127.0.0.1:8000/score_tx_id" \
  -H "Content-Type: application/json" \
  -d '{
        "tx_id": 1772,
        "threshold": 0.07093,
        "explain": true
      }'
```
Response (example)
```bash
{
  "tx_id": 1772,
  "fraud_probability": 0.158601,
  "threshold": 0.07093,
  "decision": "block",
  "top_factors": {
    "amount": 2.6046,
    "merchant_risk": 1.7103,
    "account_age_days": 1.6781
  }
}
    }'
```
Batch scoring (3 IDs + explanations)
```bash
curl -X POST "http://127.0.0.1:8000/score_many" \
  -H "Content-Type: application/json" \
  -d '{
        "tx_ids": [1772, 2332, 2631],
        "threshold": 0.07093,
        "explain": true
      }'

```
## 7. Key Results & Metrics
(Validation on hold-out set; fraud rate ~5–6%)

- LightGBM (best run):

  - PR-AUC: ~0.07

  - Recall @ 1% FPR: ~1.6–2.0% (best seen ~1.99%)

  - Cost-aware threshold: ~0.07093 (guardrail ≤1% FPR)

- XGBoost (baseline):

  - Similar PR-AUC, slightly lower recall @ 1% FPR in our runs

  Numbers vary slightly across seeds/splits; the repo stores val_scores.csv for the last run and uses .env:THRESHOLD for serving.

Business framing: We optimize to catch more fraud while keeping false positive reviews low (≤1% FPR), and expose the calibrated probability for downstream risk policies.

## 8. Model Explainability (SHAP)

For transparency:

- The API can attach top feature contributions per prediction ("explain": true).

- Under the hood, we use the base model from the bundle with shap.TreeExplainer.

- Explanations are local (per transaction) and ranked by absolute impact.

When explanations are not needed, set "explain": false to minimize latency.

## 9. Versioning & Reproducibility

- Bundle export: artifacts/model_v12.pkl = calibrated model + base model

- Locked columns: artifacts/feature_columns.json guarantees serving column order

- Feature schema: artifacts/feature_config.json documents numeric/categorical splits

- Metrics snapshot: artifacts/val_scores.csv read by /metrics

## 10. Security & Secrets

- Never put real credentials in code or README.

- Commit .env.example only.

- Real secrets live in local .env (git-ignored).

- DB access uses a read-only user (fraud_ro) scoped to fraud_radar.

## 11. Project Structure
```bash
Fraud-Radar/
├─ artifacts/                    # exported models & metadata (git-ignored)
│  ├─ model_v12.pkl             # calibrated bundle (with base model for SHAP)
│  ├─ feature_columns.json      # exact training order after one-hot
│  └─ feature_config.json       # {"num_feats": [...], "cat_feats": [...]}
├─ data/
│  └─ raw/                      # CSVs for DB load (git-ignored)
├─ src/
│  ├─ api/
│  │  └─ app.py                 # FastAPI app (uses .env)
│  ├─ models/
│  │  ├─ train_lgbm_v11.py      # LightGBM training
│  │  ├─ train_xgb_v12.py       # XGBoost baseline
│  │  ├─ pick_threshold.py      # cost-aware threshold search
│  │  └─ export_model.py        # bundle exporter (if retrained)
│  └─ sql/                      # optional SQL helpers
├─ .env.example                 # template for secrets
├─ .gitignore
└─ README.md
```
## 12. Future Improvements

- Data scale-up (more merchants/devices/IPs) and richer networks (graph features)

- Online learning / retraining cadence with model monitoring

- Drift alerts (feature, label, performance) and champion/challenger setup

- Auth & audit on the API (keys, rate limits, request logging)

- Docker & CI/CD pipeline for reproducible deploys

## Built With
MySQL 8 | Python 3.10 | pandas | scikit-learn | LightGBM | XGBoost | SHAP | FastAPI | Uvicorn | python-dotenv

## License
MIT
