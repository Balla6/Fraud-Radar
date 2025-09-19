# Fraud-Radar

End-to-end **card-fraud detection** project designed to feel like a real production system:

- SQL features in **MySQL** (card/device/IP/merchant + rolling windows)
- **LightGBM** model (XGBoost baseline), **probability calibration**
- **Cost-aware thresholding** (operate around ~**1% FPR**)
- **FastAPI** microservice with single & batch scoring and optional **SHAP** explanations
- **Exported bundle** (model + locked column list) for serving parity
- Secrets via **`.env`** (no passwords in code or requests)

> ⚠️ **Security note**
>
> - Commit `.env.example` with placeholders  
> - **Do NOT commit `.env`** with real credentials (already ignored by `.gitignore`)

---

## Table of contents

- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [API](#api)
- [Training & export](#training--export)
- [Data assumptions](#data-assumptions)
- [Metrics & operating point](#metrics--operating-point)
- [Reproducibility & safety](#reproducibility--safety)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)
- [Credits](#credits)

---

## Project structure

fraud-radar/
├─ artifacts/ # exported models & metadata (generated)
│ ├─ model_v12.pkl # calibrated sklearn model bundle (LightGBM/XGB)
│ ├─ feature_columns.json # exact training column order (after dummies)
│ └─ feature_config.json # {"num_feats": [...], "cat_feats": [...]}
├─ data/
│ └─ raw/ # generated CSVs (ignored by git)
├─ src/
│ ├─ api/
│ │ └─ app.py # FastAPI app (score_tx_id / score_many / metrics)
│ ├─ models/
│ │ ├─ train_lgbm_v11.py
│ │ ├─ train_xgb_v12.py
│ │ ├─ pick_threshold.py
│ │ └─ export_model.py
│ └─ ingest.py # (optional) synthesize CSVs or load data
├─ .env.example # sample env file (safe to commit)
├─ requirements.txt
└─ README.md

---
## Quick start

### 1) Requirements

- Python **3.10**
- MySQL **8.x**
- Git Bash/PowerShell (Windows) or Bash (macOS/Linux)

```bash
# create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

2) MySQL setup
1. Copy the example env to a real one locally:


```bash
Copy code
cp .env.example .env
```

2. Edit .env on your machine (do not commit this file):

- Replace DB_PASS=change_me with your local MySQL password.

- Keep .env out of git (the repo’s .gitignore already does this).

.env.example (committed)

```bash
DB_HOST=127.0.0.1
DB_NAME=fraud_radar
DB_USER=fraud_ro
DB_PASS=change_me
THRESHOLD=0.07093
```
The API loads .env via python-dotenv. Real secrets never go in the README or in the repo.

3) Start the API
bash
Copy code
uvicorn src.api.app:app --reload
# Docs: http://127.0.0.1:8000/docs
Health check:

bash
Copy code
curl http://127.0.0.1:8000/health
API
POST /score_tx_id
Score a single transaction already present in train_data_v11 (MySQL).

Request

json
Copy code
{
  "tx_id": 1772,
  "threshold": 0.07093,
  "explain": true
}
Response

json
Copy code
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
POST /score_many
Batch score by IDs (deduped; original order preserved).

Request

json
Copy code
{
  "tx_ids": [1772, 2332, 2631],
  "threshold": 0.07093,
  "explain": true
}
Response

json
Copy code
{
  "threshold": 0.07093,
  "results": [
    { "tx_id": 1772, "fraud_probability": 0.158601, "decision": "block", "top_factors": { "amount": 2.6046, "merchant_risk": 1.7103, "account_age_days": 1.6781 } },
    { "tx_id": 2332, "fraud_probability": 0.042118, "decision": "allow", "top_factors": { "merchant_risk": 1.9062, "amount": 1.1201, "amt_over_card_avg30": 0.8823 } },
    { "tx_id": 2631, "fraud_probability": 0.083412, "decision": "block", "top_factors": { "merchant_risk": 2.0203, "amount": 1.1155, "device_tx_30d": 0.7741 } }
  ]
}
GET /metrics
Returns the last validation metrics (from artifacts/val_scores.csv) plus default threshold.

Example

json
Copy code
{
  "PR_AUC": 0.0711,
  "Recall@1%FPR": 0.0199,
  "threshold": 0.07093
}
Training & export (summary)
bash
Copy code
# Train LightGBM with feature set v11
python src/models/train_lgbm_v11.py

# Optional: XGBoost baseline
python src/models/train_xgb_v12.py

# Choose cost-aware threshold (balances FPR/recall under a simple cost model)
python src/models/pick_threshold.py

# Export calibrated model + locked column list
python src/models/export_model.py
# -> artifacts/model_v12.pkl, feature_columns.json, feature_config.json, val_scores.csv
Why calibration & locked columns?

Calibrated probabilities make thresholding meaningful (esp. at low FPR).

Reindexing to feature_columns.json ensures serving matches training even when one-hot columns shift.

Data assumptions
Transactions & features are in MySQL database fraud_radar.

Inference queries read from train_data_v11.

Features include amounts, recent activity per card/device/IP, merchant risk, simple time flags, and engineered cols like:

high_amount_flag, log_amount, amt_over_card_avg30, card_min_since_prev, card_kmh_from_prev, etc.

The API coerces DB numerics to float and applies the same preprocessing as training.

Example cURL
bash
Copy code
# single
curl -X POST "http://127.0.0.1:8000/score_tx_id" \
  -H "Content-Type: application/json" \
  -d '{"tx_id":1772,"threshold":0.07093,"explain":true}'

# batch
curl -X POST "http://127.0.0.1:8000/score_many" \
  -H "Content-Type: application/json" \
  -d '{"tx_ids":[1772,2332,2631],"threshold":0.07093,"explain":true}'
Development
bash
Copy code
# run API locally
uvicorn src.api.app:app --reload

# format (optional)
# pip install ruff black
ruff check --fix .
black .

# run tests (if/when added)
pytest -q
Secrets & repo hygiene
Do not commit .env — use .env.example in the repo, and .env only on your machine.

Use a read-only MySQL user for the API (DB_USER=fraud_ro).

If deploying, inject env vars via your platform’s secret manager.

Benchmarks (current snapshot, synthetic data)
LightGBM v11: PR_AUC ≈ 0.0711, Recall@1%FPR ≈ 0.0199

Suggested default threshold: THRESHOLD=0.07093

Expect to re-tune on real data.

License
MIT (see LICENSE).

Credits
Built with Python, MySQL, pandas, scikit-learn, LightGBM, XGBoost, FastAPI, and SHAP.

perl


