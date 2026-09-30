import json, re, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import language_tool_python

tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")
T = json.load(open("cache/transcripts.json"))
tool = language_tool_python.LanguageTool("en-US")
FILL = {"um","uh","er","ah","hmm","like","so","well","actually","basically"}

def feats(key):
    d = T[key]; text = d["text"].strip()
    words = re.findall(r"[a-zA-Z']+", text.lower()); n = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    segs = d["segments"]
    dur = (segs[-1]["end"] - segs[0]["start"]) if segs else 60
    gaps = [segs[i+1]["start"] - segs[i]["end"] for i in range(len(segs)-1)]
    errs = tool.check(text)
    gram = [e for e in errs if e.category not in ("TYPOS", "STYLE")]
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return dict(
        n_words=n, n_sents=len(sents), wps=n/max(len(sents),1),
        ttr=len(set(words))/n, avg_wlen=np.mean([len(w) for w in words]) if words else 0,
        filler_rate=sum(w in FILL for w in words)/n, rep_rate=reps/n,
        err_rate=len(errs)/n*100, gram_err_rate=len(gram)/n*100,
        wpm=n/max(dur,1)*60, mean_gap=np.mean(gaps) if gaps else 0,
        long_pauses=sum(g > 1.0 for g in gaps))

def build(df, split):
    return pd.DataFrame([feats(f"{split}/{k}") for k in df.filename])

Xtr, Xte, y = build(tr, "train"), build(te, "test"), tr.label.values
Xtr.to_csv("cache/feat_train.csv", index=False)
Xte.to_csv("cache/feat_test.csv", index=False)

model = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10))
bins = np.digitize(y, [2.25, 2.75, 3.25, 3.75])
oof = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, bins):
    oof[b] = model().fit(Xtr.iloc[a], y[a]).predict(Xtr.iloc[b])
rmse = lambda t, p: mean_squared_error(t, p) ** 0.5
print(f"Mean-baseline RMSE : {rmse(y, np.full_like(y, y.mean())):.3f}")
print(f"CV RMSE            : {rmse(y, np.clip(oof,0,5)):.3f}   CV Pearson: {pearsonr(y, oof)[0]:.3f}")

final = model().fit(Xtr, y)
print(f"TRAIN RMSE (required in notebook): {rmse(y, np.clip(final.predict(Xtr),0,5)):.3f}")
pd.DataFrame({"filename": te.filename,
              "label": np.clip(final.predict(Xte), 0, 5)}).to_csv("submission.csv", index=False)
print("saved submission.csv")