# SHL Audio Grammar Scoring Challenge

Predicts a 0–5 grammar score for spoken audio clips.

## Approach
1. Transcribe audio with Whisper (`mlx-whisper`) — `transcribe.py`
2. Extract features:
   - Text stats + LanguageTool grammar errors — `baseline.py`
   - CoLA grammatical-acceptability + GPT-2 perplexity — `features_v2.py`
   - spaCy syntax features (tested, did not improve CV — excluded from final model) — `features_v3.py`
   - Whisper encoder audio embeddings (PCA-reduced) — `features_v4.py`
3. Model: Ridge regression, 5-fold stratified CV, PCA/alpha swept — `baseline_v4.py`

## Results (5-fold CV)
| Model | CV RMSE | CV Pearson |
|---|---|---|
| Mean baseline | 0.766 | — |
| Text features only | 0.684 | 0.450 |
| + CoLA + perplexity | 0.646 | 0.540 |
| + spaCy syntax (rejected, overfit) | 0.650 | 0.530 |
| **+ audio embeddings (final)** | **0.537** | **0.713** |

Final model: text + CoLA/perplexity features + Whisper audio embeddings (PCA=70) + Ridge(alpha=40).
Training RMSE: 0.437.

Kaggle leaderboard: Public RMSE 0.550, Private RMSE 0.577.

## Setup
\`\`\`
pip install -r requirements.txt
\`\`\`

## Run (in order)
\`\`\`
python transcribe.py       # transcribe all audio (cached to cache/transcripts.json)
python baseline.py         # baseline text features -> cache/feat_*.csv
python features_v2.py      # CoLA + perplexity features -> cache/feat_*_v2.csv
python features_v3.py      # spaCy syntax features (evaluated, not used in final model)
python features_v4.py      # Whisper audio embeddings -> cache/audio_emb_*.npy
python baseline_v4.py      # final model: sweeps PCA/alpha, trains, writes submission.csv
\`\`\`
