import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")
y = tr.label.values

def load(split):
    return pd.concat([
        pd.read_csv(f"cache/feat_{split}.csv"),
        pd.read_csv(f"cache/feat_{split}_v2.csv"),
        pd.read_csv(f"cache/feat_{split}_v3.csv"),
    ], axis=1)

Xtr, Xte = load("train"), load("test")
Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(Xtr.median())
Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(Xtr.median())

rmse = lambda t, p: mean_squared_error(t, p) ** 0.5
bins = np.digitize(y, [2.25, 2.75, 3.25, 3.75])

def cv_score(alpha):
    oof = np.zeros(len(y))
    for a_idx, b_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, bins):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(Xtr.iloc[a_idx], y[a_idx])
        oof[b_idx] = m.predict(Xtr.iloc[b_idx])
    return rmse(y, np.clip(oof, 0, 5)), pearsonr(y, oof)[0]

print("v2 (before spaCy) was RMSE 0.646 / Pearson 0.540\n")
print("alpha sweep on v3 features (text + CoLA/ppl + spaCy):")
results = []
for alpha in [5, 10, 20, 40, 80, 150, 300]:
    r, p = cv_score(alpha)
    results.append((alpha, r, p))
    print(f"  alpha={alpha:<5} CV RMSE={r:.3f}  CV Pearson={p:.3f}")

best_alpha = min(results, key=lambda x: x[1])[0]
best_rmse, best_pearson = min(results, key=lambda x: x[1])[1:]
print(f"\nBest alpha: {best_alpha}  ->  CV RMSE={best_rmse:.3f}  CV Pearson={best_pearson:.3f}")

final = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(Xtr, y)
train_rmse = rmse(y, np.clip(final.predict(Xtr), 0, 5))
print(f"TRAIN RMSE (best alpha): {train_rmse:.3f}")

pd.DataFrame({"filename": te.filename,
              "label": np.clip(final.predict(Xte), 0, 5)}).to_csv("submission.csv", index=False)
print("saved submission.csv")