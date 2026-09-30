import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")
y = tr.label.values

# text + CoLA/perplexity features (v2 — our best so far)
Xtr_txt = pd.concat([pd.read_csv("cache/feat_train.csv"), pd.read_csv("cache/feat_train_v2.csv")], axis=1)
Xte_txt = pd.concat([pd.read_csv("cache/feat_test.csv"), pd.read_csv("cache/feat_test_v2.csv")], axis=1)
Xtr_txt = Xtr_txt.replace([np.inf, -np.inf], np.nan).fillna(Xtr_txt.median())
Xte_txt = Xte_txt.replace([np.inf, -np.inf], np.nan).fillna(Xtr_txt.median())

# audio embeddings
Etr = np.load("cache/audio_emb_train.npy")
Ete = np.load("cache/audio_emb_test.npy")

rmse = lambda t, p: mean_squared_error(t, p) ** 0.5
bins = np.digitize(y, [2.25, 2.75, 3.25, 3.75])

def cv_score(n_pca, alpha):
    oof = np.zeros(len(y))
    for a_idx, b_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr_txt, bins):
        # fit PCA only on this fold's training audio embeddings (avoid leakage)
        pca = PCA(n_components=n_pca, random_state=42).fit(Etr[a_idx])
        Atr_pca = pca.transform(Etr[a_idx])
        Ava_pca = pca.transform(Etr[b_idx])

        Xa = np.hstack([Xtr_txt.iloc[a_idx].values, Atr_pca])
        Xb = np.hstack([Xtr_txt.iloc[b_idx].values, Ava_pca])

        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(Xa, y[a_idx])
        oof[b_idx] = m.predict(Xb)
    return rmse(y, np.clip(oof, 0, 5)), pearsonr(y, oof)[0]

print("v2 (text + CoLA/ppl, no audio) was RMSE 0.646 / Pearson 0.540\n")
print("sweeping PCA components x alpha:")
results = []
for n_pca in [50, 70, 90, 120, 150]:
    for alpha in [10, 20, 40, 80]:
        r, p = cv_score(n_pca, alpha)
        results.append((n_pca, alpha, r, p))
        print(f"  pca={n_pca:<4} alpha={alpha:<4} CV RMSE={r:.3f}  CV Pearson={p:.3f}")

best = min(results, key=lambda x: x[2])
print(f"\nBest: pca={best[0]} alpha={best[1]} -> CV RMSE={best[2]:.3f}  CV Pearson={best[3]:.3f}")

# refit best config on full training data
n_pca, alpha = best[0], best[1]
pca_final = PCA(n_components=n_pca, random_state=42).fit(Etr)
Atr_pca = pca_final.transform(Etr)
Ate_pca = pca_final.transform(Ete)
Xtr_final = np.hstack([Xtr_txt.values, Atr_pca])
Xte_final = np.hstack([Xte_txt.values, Ate_pca])

final = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr_final, y)
train_rmse = rmse(y, np.clip(final.predict(Xtr_final), 0, 5))
print(f"TRAIN RMSE: {train_rmse:.3f}")

pd.DataFrame({"filename": te.filename,
              "label": np.clip(final.predict(Xte_final), 0, 5)}).to_csv("submission.csv", index=False)
print("saved submission.csv")