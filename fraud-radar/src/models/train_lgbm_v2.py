from getpass import getpass
import pandas as pd, numpy as np
import mysql.connector as mc
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_curve

pw = getpass("MySQL password for user 'root': ")
conn = mc.connect(host="localhost", user="root", password=pw, database="fraud_radar")

q = """
SELECT tx_id, tx_time, amount,
       card_tx_24h, card_amt_24h, card_avg_24h,
       device_card_count_7d,
       merch_chargeback_rate_30d,
       merchant_risk,
       is_fraud
FROM train_data_v2
ORDER BY tx_time
"""
df = pd.read_sql(q, conn, parse_dates=["tx_time"])
conn.close()

feats = ["amount","card_tx_24h","card_amt_24h","card_avg_24h",
         "device_card_count_7d","merch_chargeback_rate_30d","merchant_risk"]
df[feats] = df[feats].fillna(0)
y = df["is_fraud"].astype(int).values

split = int(len(df)*0.8)
Xtr, ytr = df.loc[:split-1, feats], y[:split]
Xte, yte = df.loc[split:, feats], y[split:]

pos, neg = max(1, np.sum(ytr==1)), max(1, np.sum(ytr==0))
spw = neg / pos

model = LGBMClassifier(
    n_estimators=800, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    max_depth=-1, min_child_samples=40,
    scale_pos_weight=spw, random_state=7
)
model.fit(Xtr, ytr)
proba = model.predict_proba(Xte)[:,1]

from sklearn.calibration import CalibratedClassifierCV
# quick isotonic calibration for better thresholds
cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
cal.fit(Xtr, ytr)
proba = cal.predict_proba(Xte)[:,1]

from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(yte, proba)
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(yte, proba)
recall_at_1_fpr = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0

print({"PR_AUC": round(pr_auc, 4), "Recall@1%FPR": round(recall_at_1_fpr, 4)})

out = df.loc[split:, ["tx_id","amount","is_fraud"]].copy()
out["y_proba"] = proba
out.to_csv("val_scores.csv", index=False)
print("Wrote val_scores.csv")
