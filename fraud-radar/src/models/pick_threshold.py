import pandas as pd, numpy as np

FALSE_NEG_COST_PCT = 0.80   # miss fraud -> lose 80% of amount
FALSE_POS_COST     = 2.50   # cost per wrongly blocked legit tx

df = pd.read_csv("val_scores.csv")  # tx_id, amount, is_fraud, y_proba
y = df["is_fraud"].astype(int).values
amt = df["amount"].values
p = df["y_proba"].values

approve_all_cost = FALSE_NEG_COST_PCT * amt[y == 1].sum()

# Try MANY small thresholds (down to 1e-6) + quantiles of your scores
cands = np.unique(np.concatenate([
    np.quantile(p, np.linspace(0, 0.999, 400)),
    [1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05]
]))

best = None
best_at_1fpr = None

for t in cands:
    yhat = (p >= t).astype(int)
    tp = np.sum((yhat==1)&(y==1))
    fp = np.sum((yhat==1)&(y==0))
    fn = np.sum((yhat==0)&(y==1))
    tn = np.sum((yhat==0)&(y==0))

    cost = FALSE_NEG_COST_PCT * amt[(yhat==0)&(y==1)].sum() + FALSE_POS_COST * fp
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    fpr  = fp/(fp+tn) if (fp+tn) else 0.0
    saved = float(approve_all_cost - cost)

    row = dict(threshold=float(t), cost=float(cost), fp=int(fp), fn=int(fn),
               precision=round(float(prec),4), recall=round(float(rec),4),
               fpr=round(float(fpr),4), saved=round(saved,2))

    if best is None or row["cost"] < best["cost"]:
        best = row
    if fpr <= 0.01 and (best_at_1fpr is None or row["recall"] > best_at_1fpr["recall"]):
        best_at_1fpr = row

print("Best threshold by expected cost:", best)
print("Best threshold with FPR <= 1%:", best_at_1fpr)
