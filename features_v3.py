import json, numpy as np, pandas as pd, spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")
T = json.load(open("cache/transcripts.json"))

def dep_depth(sent):
    def depth(tok):
        return 1 + max([depth(c) for c in tok.children], default=0)
    root = [t for t in sent if t.head == t]
    return depth(root[0]) if root else 0

def feats_v3(key):
    doc = nlp(T[key]["text"].strip())
    sents = list(doc.sents)
    n_sents = max(len(sents), 1)
    depths = [dep_depth(s) for s in sents]
    pos_counts = doc.count_by(spacy.attrs.POS)
    total_tok = max(sum(pos_counts.values()), 1)
    # incomplete sentence heuristic: no verb (ROOT that's a VERB/AUX) in the sentence
    incomplete = sum(1 for s in sents if not any(t.pos_ in ("VERB","AUX") for t in s))
    subordinate = sum(1 for t in doc if t.dep_ in ("advcl","ccomp","xcomp","acl","relcl"))
    coordinate = sum(1 for t in doc if t.dep_ == "conj")
    return dict(
        dep_depth_mean=np.mean(depths) if depths else 0,
        dep_depth_max=np.max(depths) if depths else 0,
        incomplete_frac=incomplete/n_sents,
        subordinate_rate=subordinate/total_tok*100,
        coordinate_rate=coordinate/total_tok*100,
        noun_frac=pos_counts.get(nlp.vocab.strings["NOUN"],0)/total_tok,
        verb_frac=pos_counts.get(nlp.vocab.strings["VERB"],0)/total_tok,
        pos_diversity=len(pos_counts)/total_tok*100,
        n_sents_spacy=n_sents)

def build(df, split):
    return pd.DataFrame([feats_v3(f"{split}/{k}") for k in tqdm(df.filename)])

Xtr3 = build(tr, "train"); Xte3 = build(te, "test")
Xtr3.to_csv("cache/feat_train_v3.csv", index=False)
Xte3.to_csv("cache/feat_test_v3.csv", index=False)
print("done")