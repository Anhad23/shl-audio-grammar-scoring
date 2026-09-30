import json, glob, os, numpy as np, pandas as pd, torch, torchaudio
from tqdm import tqdm
import mlx.core as mx
import mlx_whisper
from mlx_whisper.audio import load_audio, log_mel_spectrogram, N_FRAMES, pad_or_trim
from mlx_whisper.load_models import load_model

MODEL_PATH = "mlx-community/whisper-large-v3-turbo"
model = load_model(MODEL_PATH)

tr = pd.read_csv("dataset_new/csvs/train.csv")
te = pd.read_csv("dataset_new/csvs/test.csv")

def audio_embedding(path):
    audio = load_audio(path)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mel = pad_or_trim(mel, N_FRAMES, axis=-2)
    mel = mx.expand_dims(mel, axis=0)
    enc_out = model.encoder(mel)          # (1, T, D)
    emb = mx.mean(enc_out, axis=1)[0]     # (D,) mean-pool over time
    return np.array(emb)

def build(df, split):
    rows = []
    for fn in tqdm(df.filename):
        path = f"dataset_new/audios/{split}/{fn}.wav"
        rows.append(audio_embedding(path))
    return np.stack(rows)

Etr = build(tr, "train")
Ete = build(te, "test")
np.save("cache/audio_emb_train.npy", Etr)
np.save("cache/audio_emb_test.npy", Ete)
print("shapes:", Etr.shape, Ete.shape)
print("done")