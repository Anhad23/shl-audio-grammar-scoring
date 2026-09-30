import glob, mlx_whisper
f = sorted(glob.glob("dataset_new/audios/train/*.wav"))[0]
r = mlx_whisper.transcribe(f, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en")
print(f); print(r["text"])