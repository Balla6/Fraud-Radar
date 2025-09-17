\1from dotenv import load_dotenv
load_dotenv()
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

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_NAME = os.getenv("DB_NAME", "fraud_radar")
DB_USER = os.getenv("DB_USER", "fraud_ro")
DB_PASS = os.getenv("DB_PASS", "")

def get_conn():
    return mc.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)

app = FastAPI(title="Fraud Radar", version="0.1")

class ScoreByTxID(BaseModel):
    tx_id: int
    threshold: Optional[float] = None
    explain: bool = False

class ScoreMany(BaseModel):
    tx_ids: List[int] = Field(..., min_items=1, max_items=1000)
    threshold: Optional[float] = None
    explain: bool = False

@app.get("/health")
def health():
    return {"ok": True}

def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    # normalize types from MySQL
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)

    df["high_amount_flag"] = (df["amount"] > 200).astype(int)
    df["log_amount"] = np.log1p(df["amount"].fillna(0))
    df["amt_over_card_avg30"] = df["amount"] / (df["card_avg_amt_30d"].replace(0, np.nan))
    df["amt_over_card_avg30"] = df["amt_over_card_avg30"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["card_min_since_prev"] = df["card_min_since_prev"].fillna(1e6)
    df["card_kmh_from_prev"]  = df["card_kmh_from_prev"].fillna(0)
    df["device_age_days"]     = df["device_age_days"].clip(lower=0).fillna(0)
    df["account_age_days"]    = df["account_age_days"].clip(lower=0).fillna(0)
    X = pd.get_dummies(df[NUM + CAT], columns=CAT)
    return X.reindex(columns=COLS, fill_value=0).astype(np.float32)

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
    # fetch one
    try:
        conn = get_conn()
        q = f"SELECT {SELECT_COLUMNS} FROM train_data_v11 WHERE tx_id = %s"
        df = pd.read_sql_query(q, conn, params=(req.tx_id,))
    finally:
        try: conn.close()
        except: pass
    if df.empty:
        raise HTTPException(status_code=404, detail=f"tx_id {req.tx_id} not found")

    X = _prep_features(df)
    proba = float(CAL.predict_proba(X)[:,1][0])
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
    ids = list(dict.fromkeys(req.tx_ids))  # de-dup, preserve order
    if not ids:
        return {"threshold": th, "results": []}
    placeholders = ",".join(["%s"] * len(ids))

    try:
        conn = get_conn()
        q = f"SELECT {SELECT_COLUMNS} FROM train_data_v11 WHERE tx_id IN ({placeholders})"
        df = pd.read_sql_query(q, conn, params=tuple(ids))
    finally:
        try: conn.close()
        except: pass

    if df.empty:
        return {"threshold": th, "results": []}

    keep_order = [i for i in ids if i in set(df["tx_id"])]
    df = df.set_index("tx_id").loc[keep_order].reset_index()

    X = _prep_features(df)
    probs = CAL.predict_proba(X)[:,1]
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
            out = [{"tx_id": int(tx),
                    "fraud_probability": round(float(p), 6),
                    "decision": "block" if p >= th else "allow"} for tx, p in zip(df["tx_id"], probs)]
    else:
        out = [{"tx_id": int(tx),
                "fraud_probability": round(float(p), 6),
                "decision": "block" if p >= th else "allow"} for tx, p in zip(df["tx_id"], probs)]
    return {"threshold": th, "results": out}

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


