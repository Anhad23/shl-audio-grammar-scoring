import json, glob, os
import mlx_whisper
from tqdm import tqdm

OUT = "cache/transcripts.json"
MODEL = "mlx-community/whisper-large-v3-turbo"
files = sorted(glob.glob("dataset_new/audios/*/*.wav"))
done = json.load(open(OUT)) if os.path.exists(OUT) else {}

for i, f in enumerate(tqdm(files)):
    split = os.path.basename(os.path.dirname(f))            # "train" or "test"
    key = f"{split}/{os.path.splitext(os.path.basename(f))[0]}"   # e.g. "test/audio_127"
    if key in done: continue
    r = mlx_whisper.transcribe(
        f, path_or_hf_repo=MODEL, language="en",
        initial_prompt="Um, so, uh, I mean, like, you know, well...",
        condition_on_previous_text=False)
    done[key] = {"text": r["text"],
                 "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]}
                              for s in r["segments"]]}
    if i % 10 == 0: json.dump(done, open(OUT, "w"))
json.dump(done, open(OUT, "w"))
print(len(done), "transcribed; expected 606")