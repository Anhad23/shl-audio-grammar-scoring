import json, re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"

tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")
T = json.load(open("cache/transcripts.json"))

# --- CoLA model: outputs P(grammatical) per sentence ---
cola_tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-CoLA")
cola_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-CoLA").to(device).eval()

# --- GPT-2 for perplexity ---
gpt_tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

def split_sents(text):
    s = [x.strip() for x in re.split(r"[.!?]+", text) if len(x.strip().split()) >= 3]
    return s if s else [text.strip()]

@torch.no_grad()
def cola_scores(sents):
    scores = []
    for s in sents:
        ids = cola_tok(s, return_tensors="pt", truncation=True, max_length=64).to(device)
        p = torch.softmax(cola_model(**ids).logits, dim=-1)[0]
        scores.append(p[1].item())  # P(acceptable)
    return scores

@torch.no_grad()
def perplexity(text):
    ids = gpt_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if ids.input_ids.shape[1] < 2: return 100.0
    loss = gpt_model(**ids, labels=ids.input_ids).loss
    return torch.exp(loss).item()

def feats_v2(key):
    text = T[key]["text"].strip()
    sents = split_sents(text)
    c = cola_scores(sents)
    return dict(
        cola_mean=np.mean(c), cola_min=np.min(c), cola_std=np.std(c),
        cola_frac_bad=np.mean(np.array(c) < 0.5),
        ppl=min(perplexity(text), 1000),
        ppl_per_sent=np.mean([min(perplexity(s), 1000) for s in sents]))

def build(df, split):
    rows = [feats_v2(f"{split}/{k}") for k in tqdm(df.filename)]
    return pd.DataFrame(rows)

Xtr2 = build(tr, "train"); Xte2 = build(te, "test")
Xtr2.to_csv("cache/feat_train_v2.csv", index=False)
Xte2.to_csv("cache/feat_test_v2.csv", index=False)
print("done")