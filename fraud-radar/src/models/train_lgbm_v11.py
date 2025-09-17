from getpass import getpass
import pandas as pd, numpy as np
import mysql.connector as mc
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_curve

pw = getpass("MySQL password for user 'root': ")
conn = mc.connect(host="localhost", user="root", password=pw, database="fraud_radar")

q = """
SELECT
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
  merchant_category, device_os,
  is_fraud
FROM train_data_v10
ORDER BY tx_time
"""
df = pd.read_sql(q, conn, parse_dates=["tx_time"])
conn.close()

# numeric features
df["high_amount_flag"] = (df["amount"] > 200).astype(int)
df["log_amount"] = np.log1p(df["amount"].fillna(0))
df["amt_over_card_avg30"] = df["amount"] / (df["card_avg_amt_30d"].replace(0, np.nan))
df["amt_over_card_avg30"] = df["amt_over_card_avg30"].replace([np.inf, -np.inf], np.nan).fillna(0)
df["card_min_since_prev"] = df["card_min_since_prev"].fillna(1e6)
df["card_kmh_from_prev"]  = df["card_kmh_from_prev"].fillna(0)
df["device_age_days"]     = df["device_age_days"].clip(lower=0).fillna(0)
df["account_age_days"]    = df["account_age_days"].clip(lower=0).fillna(0)

num_feats = [
  "amount","log_amount","high_amount_flag","amt_over_card_avg30",
  "card_tx_24h","card_amt_24h","card_avg_24h",
  "device_card_count_7d","device_tx_24h",
  "device_tx_30d","device_prior_fraud_30d","device_prior_fraud_rate_30d",
  "merch_chargeback_rate_30d","merchant_risk",
  "night_time","country_mismatch",
  "card_min_since_prev","card_kmh_from_prev","impossible_speed",
  "device_age_days","account_age_days",
  "ip_tx_30d","ip_prior_fraud_30d","ip_prior_fraud_rate_30d",
  "ip_card_count_7d","new_device_for_card_30d"
]
cat_feats = ["merchant_category","device_os"]

y = df["is_fraud"].astype(int).values

# time split
split = int(len(df)*0.8)
train = df.iloc[:split].copy()
test  = df.iloc[split:].copy()

# one-hot on train, then align test
Xtr = pd.get_dummies(train[num_feats + cat_feats], columns=cat_feats)
Xte = pd.get_dummies(test [num_feats + cat_feats], columns=cat_feats)
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

ytr, yte = y[:split], y[split:]

# imbalance handling
pos, neg = max(1, (ytr==1).sum()), max(1, (ytr==0).sum())
spw = neg / pos

model = LGBMClassifier(
    n_estimators=2200, learning_rate=0.035,
    subsample=0.85, colsample_bytree=0.9,
    num_leaves=64, min_child_samples=25,
    scale_pos_weight=spw, random_state=7
)
model.fit(Xtr, ytr)

# calibrate for thresholding (keeps ranking)
cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
cal.fit(Xtr, ytr)
proba = cal.predict_proba(Xte)[:,1]

pr_auc = average_precision_score(yte, proba)
fpr, tpr, _ = roc_curve(yte, proba)
recall_at_1_fpr = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0

print({"PR_AUC": round(pr_auc, 4), "Recall@1%FPR": round(recall_at_1_fpr, 4)})

out = test[["tx_id","amount","is_fraud"]].copy()
out["y_proba"] = proba
out.to_csv("val_scores.csv", index=False)
print("Wrote val_scores.csv")
