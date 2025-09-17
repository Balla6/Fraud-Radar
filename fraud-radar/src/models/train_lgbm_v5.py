from getpass import getpass
import pandas as pd, numpy as np
import mysql.connector as mc
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_curve

pw = getpass("MySQL password for user 'root': ")
conn = mc.connect(host="localhost", user="root", password=pw, database="fraud_radar")

q = """
SELECT tx_id, tx_time, amount,
       card_tx_24h, card_amt_24h, card_avg_24h,
       device_card_count_7d, device_tx_24h,
       device_tx_30d, device_prior_fraud_30d, device_prior_fraud_rate_30d,
       merch_chargeback_rate_30d, merchant_risk,
       card_avg_amt_30d, night_time, country_mismatch,
       is_fraud
FROM train_data_v5
ORDER BY tx_time
"""
df = pd.read_sql(q, conn, parse_dates=["tx_time"])
conn.close()

df["log_amount"] = np.log1p(df["amount"].fillna(0))
df["amt_over_card_avg30"] = df["amount"] / (df["card_avg_amt_30d"].replace(0, np.nan))
df["amt_over_card_avg30"] = df["amt_over_card_avg30"].replace([np.inf, -np.inf], np.nan).fillna(0)

feats = [
    "amount","log_amount","amt_over_card_avg30",
    "card_tx_24h","card_amt_24h","card_avg_24h",
    "device_card_count_7d","device_tx_24h",
    "device_tx_30d","device_prior_fraud_30d","device_prior_fraud_rate_30d",
    "merch_chargeback_rate_30d","merchant_risk",
    "night_time","country_mismatch"
]
df[feats] = df[feats].fillna(0)
y = df["is_fraud"].astype(int).values

split = int(len(df)*0.8)
Xtr, ytr = df.loc[:split-1, feats], y[:split]
Xte, yte = df.loc[split:, feats], y[split:]

pos, neg = max(1, (ytr==1).sum()), max(1, (ytr==0).sum())
spw = neg / pos

model = LGBMClassifier(
    n_estimators=1200, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.9,
    max_depth=-1, min_child_samples=30,
    scale_pos_weight=spw, random_state=7
)
model.fit(Xtr, ytr)

cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
cal.fit(Xtr, ytr)
proba = cal.predict_proba(Xte)[:,1]

pr_auc = average_precision_score(yte, proba)
fpr, tpr, _ = roc_curve(yte, proba)
recall_at_1_fpr = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0

print({"PR_AUC": round(pr_auc, 4), "Recall@1%FPR": round(recall_at_1_fpr, 4)})

out = df.loc[split:, ["tx_id","amount","is_fraud"]].copy()
out["y_proba"] = proba
out.to_csv("val_scores.csv", index=False)
print("Wrote val_scores.csv")
