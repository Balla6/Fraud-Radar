from getpass import getpass
import pandas as pd
import numpy as np
import mysql.connector as mc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve

# ask for MySQL password so we don't hardcode it
pw = getpass("MySQL password for user 'root': ")

conn = mc.connect(
    host="localhost",
    user="root",
    password=pw,
    database="fraud_radar"
)

# pull data ordered by time for a leakage-safe split
q = """
SELECT tx_id, tx_time, amount,
       card_tx_24h, card_amt_24h, card_avg_24h,
       device_card_count_7d,
       merchant_risk,
       is_fraud
FROM train_data
ORDER BY tx_time
"""
df = pd.read_sql(q, conn, parse_dates=["tx_time"])
conn.close()

# features + target
features = ["amount","card_tx_24h","card_amt_24h","card_avg_24h",
            "device_card_count_7d","merchant_risk"]
df[features] = df[features].fillna(0)
y = df["is_fraud"].astype(int).values

# time-based split (80% train, 20% test)
split = int(len(df) * 0.8)
X_train, y_train = df.loc[:split-1, features], y[:split]
X_test,  y_test  = df.loc[split:, features], y[split:]

# quick baseline: logistic regression with class_weight (handles imbalance)
model = LogisticRegression(max_iter=200, class_weight="balanced")
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]

# metrics
pr_auc = average_precision_score(y_test, scores)
fpr, tpr, thr = roc_curve(y_test, scores)
recall_at_1_fpr = float(tpr[fpr <= 0.01].max()) if np.any(fpr <= 0.01) else 0.0

print({"PR_AUC": round(pr_auc, 4), "Recall@1%FPR": round(recall_at_1_fpr, 4)})

# save scores for cost-based thresholding next
out = df.loc[split:, ["tx_id","amount","is_fraud"]].copy()
out["y_proba"] = scores
out.to_csv("val_scores.csv", index=False)
print("Wrote val_scores.csv in the project folder.")
