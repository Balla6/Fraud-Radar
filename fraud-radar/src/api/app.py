# src/api/app.py
import os, json, logging, numpy as np, pandas as pd
import joblib, mysql.connector as mc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fraud-radar")

ART_DIR = "artifacts"
bundle = joblib.load(os.path.join(ART_DIR, "model_v12.pkl"))
CAL   = bundle["calibrated_model"]
BASE  = bundle.get("base_model", None)  # for SHAP explanations
with open(os.path.join(ART_DIR, "feature_columns.json")) as f: COLS = json.load(f)
with open(os.path.join(ART_DIR, "feature_config.json"))  as f: CFG  = json.load(f)
NUM, CAT = CFG["num_feats"], CFG["cat_feats"]

DEFAULT_TH = float(os.getenv("THRESHOLD", "0.07093"))

app = FastAPI(title="Fraud Radar", version="0.1")

class ScoreByTxID(BaseModel):
    tx_id: int
    db_password: str
    threshold: Optional[float] = None
    explain: bool = False

class ScoreMany(BaseModel):
    tx_ids: List[int] = Field(..., min_items=1, max_items=1000)
    db_password: str
    threshold: Optional[float] = None
    explain: bool = False

@app.get("/health")
def health():
    return {"ok": True}

def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, build derived columns, align to training columns."""
    # 1) Coerce numeric features to float (handles Decimal from MySQL)
    for c in NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Coerce categoricals to string
    for c in CAT:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("NA")

    # 3) Derivations (now safe to use NumPy ufuncs)
    df["amount"] = df["amount"].fillna(0.0)
    df["high_amount_flag"] = (df["amount"].astype(float) > 200.0).astype(int)
    df["log_amount"] = np.log1p(df["amount"].astype(float))

    card_avg30 = pd.to_numeric(df["card_avg_amt_30d"], errors="coerce").replace(0, np.nan)
    df["amt_over_card_avg30"] = (df["amount"].astype(float) / card_avg30).replace([np.inf, -np.inf], np.nan)

    df["card_min_since_prev"] = pd.to_numeric(df["card_min_since_prev"], errors="coerce").fillna(1e6)
    df["card_kmh_from_prev"]  = pd.to_numeric(df["card_kmh_from_prev"], errors="coerce").fillna(0.0)
    df["device_age_days"]     = pd.to_numeric(df["device_age_days"], errors="coerce").clip(lower=0).fillna(0)
    df["account_age_days"]    = pd.to_numeric(df["account_age_days"], errors="coerce").clip(lower=0).fillna(0)

    # 4) One-hot + align
    X = pd.get_dummies(df[NUM + CAT], columns=CAT)
    X = X.reindex(columns=COLS, fill_value=0).astype(np.float32)
    return X

SELECT_COLUMNS = """
  tx_id, tx_time, amount,
  card_tx_24h, card_amt_24h, card_avg_24h,
  device_card_count_7d, device_tx_24h,
  device_tx_30d, device_prior_fraud_30d, device_prior_fraud_rate_30d,
  merch_chargeback_rate_30d, merchant_risk,
  card_avg_amt_30d, night_time, country_mismatch,
  card_min_since_prev, card_kmh_from_prev, impossible_speed,
  device_age_days, account_age_days,
  ip_tx_30d, ip_prior_fraud_30d, ip_prior_fraud_rate_30d,
  ip_card_count_7d, new_device_for_card_30d,
  merchant_category, device_os
"""

@app.post("/score_tx_id")
def score_tx_id(req: ScoreByTxID):
    th = req.threshold if req.threshold is not None else DEFAULT_TH
    try:
        conn = mc.connect(host="localhost", user="root", password=req.db_password, database="fraud_radar")
        q = f"SELECT {SELECT_COLUMNS} FROM train_data_v11 WHERE tx_id = %s"
        df = pd.read_sql_query(q, conn, params=(req.tx_id,))
    finally:
        try: conn.close()
        except: pass
    if df.empty:
        raise HTTPException(status_code=404, detail=f"tx_id {req.tx_id} not found")

    X = _prep_features(df)
    proba = float(CAL.predict_proba(X)[:, 1][0])
    result = {
        "tx_id": int(req.tx_id),
        "fraud_probability": round(proba, 6),
        "threshold": th,
        "decision": "block" if proba >= th else "allow",
    }

    if req.explain and BASE is not None:
        try:
            import shap
            explainer = shap.TreeExplainer(BASE)
            sval = explainer.shap_values(X)
            if isinstance(sval, list): sval = sval[1]
            contrib = (pd.Series(sval[0], index=COLS)
                         .sort_values(key=lambda s: s.abs(), ascending=False)
                         .head(5).round(4).to_dict())
            result["top_factors"] = contrib
        except Exception as e:
            log.warning(f"SHAP explanation failed: {e}")
    return result

@app.post("/score_many")
def score_many(req: ScoreMany):
    th = req.threshold if req.threshold is not None else DEFAULT_TH

    # de-dup but keep order
    ids, seen = [], set()
    for t in req.tx_ids:
        ti = int(t)
        if ti not in seen:
            seen.add(ti)
            ids.append(ti)
    if not ids:
        return {"threshold": th, "results": []}

    placeholders = ",".join(["%s"] * len(ids))
    q = f"SELECT {SELECT_COLUMNS} FROM train_data_v11 WHERE tx_id IN ({placeholders})"

    try:
        conn = mc.connect(host="localhost", user="root", password=req.db_password, database="fraud_radar")
        cur = conn.cursor(dictionary=True)
        cur.execute(q, ids)
        rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception as e:
        log.exception("DB error in /score_many")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not rows:
        return {"threshold": th, "results": []}

    df = pd.DataFrame(rows)
    order = [i for i in ids if i in set(df["tx_id"].tolist())]
    df = df.set_index("tx_id").loc[order].reset_index()

    try:
        X = _prep_features(df)
        probs = CAL.predict_proba(X)[:, 1]
    except Exception as e:
        log.exception("Feature prep or prediction failed in /score_many")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    out = []
    if req.explain and BASE is not None:
        try:
            import shap
            explainer = shap.TreeExplainer(BASE)
            sval = explainer.shap_values(X)
            if isinstance(sval, list): sval = sval[1]
            for tx, p, sv in zip(df["tx_id"].tolist(), probs.tolist(), sval):
                top = (pd.Series(sv, index=COLS)
                         .sort_values(key=lambda s: s.abs(), ascending=False)
                         .head(3).round(4).to_dict())
                out.append({
                    "tx_id": int(tx),
                    "fraud_probability": round(float(p), 6),
                    "decision": "block" if p >= th else "allow",
                    "top_factors": top
                })
        except Exception as e:
            log.warning(f"SHAP explanation failed (batch): {e}")
            out = [{
                "tx_id": int(tx),
                "fraud_probability": round(float(p), 6),
                "decision": "block" if p >= th else "allow"
            } for tx, p in zip(df["tx_id"], probs)]
    else:
        out = [{
            "tx_id": int(tx),
            "fraud_probability": round(float(p), 6),
            "decision": "block" if p >= th else "allow"
        } for tx, p in zip(df["tx_id"], probs)]

    missing = [i for i in ids if i not in set(df["tx_id"])]
    resp = {"threshold": th, "results": out}
    if missing:
        resp["missing_tx_ids"] = missing
    return resp

@app.get("/metrics")
def metrics():
    path = os.path.join(ART_DIR, "val_scores.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="val_scores.csv not found")
    try:
        df = pd.read_csv(path)
        last = df.tail(1).to_dict(orient="records")[0]
        last["threshold"] = DEFAULT_TH
        return last
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}")
