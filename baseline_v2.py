import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")
y = tr.label.values

Xtr = pd.concat([pd.read_csv("cache/feat_train.csv"), pd.read_csv("cache/feat_train_v2.csv")], axis=1)
Xte = pd.concat([pd.read_csv("cache/feat_test.csv"), pd.read_csv("cache/feat_test_v2.csv")], axis=1)
Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(Xtr.median())
Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(Xtr.median())

model = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10))
bins = np.digitize(y, [2.25, 2.75, 3.25, 3.75])
oof = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, bins):
    oof[b] = model().fit(Xtr.iloc[a], y[a]).predict(Xtr.iloc[b])
rmse = lambda t, p: mean_squared_error(t, p) ** 0.5
print(f"v1 (text-only) baseline was RMSE 0.684 / Pearson 0.450")
print(f"v2 CV RMSE: {rmse(y, np.clip(oof,0,5)):.3f}   CV Pearson: {pearsonr(y, oof)[0]:.3f}")

final = model().fit(Xtr, y)
print(f"TRAIN RMSE: {rmse(y, np.clip(final.predict(Xtr),0,5)):.3f}")
pd.DataFrame({"filename": te.filename, "label": np.clip(final.predict(Xte),0,5)}).to_csv("submission.csv", index=False)