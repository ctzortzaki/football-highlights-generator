import os
import cv2
import librosa
import whisper
import numpy as np
import ffmpeg
import json
from tqdm import tqdm
import torch
import pandas as pd
import csv

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = whisper.load_model("medium", device=DEVICE)

# ==== AUDIO ====

def extract_audio(video_path, out_path="audio.wav"):
    ffmpeg.input(video_path).output(out_path, ac=1, ar='16000').run(overwrite_output=True)
    return out_path

def extract_audio_features(audio_path, csv_path="audio_features.csv", interval_sec=2):
    y, sr = librosa.load(audio_path)
    hop_length = 512
    energy = librosa.feature.rms(y=y, hop_length=hop_length).flatten()
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).flatten()
    frame_times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    df = pd.DataFrame({
        "time": frame_times,
        "energy": energy,
        "spectral_flux": flux
    })

    # Bin data into 2-second intervals
    df["interval"] = (df["time"] // interval_sec).astype(int)
    # Aggregate (mean per interval)
    df_agg = df.groupby("interval").agg({
        "time": "first",
        "energy": "mean",
        "spectral_flux": "mean"
    }).reset_index(drop=True)
    df_agg.to_csv(csv_path, index=False)

# ==== TEXT ====

def transcribe_audio(audio_path, csv_path="transcript.csv", interval_sec=2):
    result = WHISPER_MODEL.transcribe(audio_path)
    segments = result.get("segments", [])

    # Get total duration to define number of intervals
    if segments:
        total_duration = segments[-1]["end"]
    else:
        total_duration = 0
    num_intervals = int(np.ceil(total_duration / interval_sec))
    interval_data = []
    for i in range(num_intervals):
        start_time = i * interval_sec
        end_time = (i + 1) * interval_sec
        text_list = []
        confidences = []

        for seg in segments:
            # If segment overlaps with interval
            if seg["end"] > start_time and seg["start"] < end_time:
                text_list.append(seg["text"])
                # if "confidence" in seg:
                #     confidences.append(seg["confidence"])
        if text_list:
            entry = {
                "start_time": start_time,
                # "end_time": end_time,
                "text": " ".join(text_list),
                # "confidence": np.mean(confidences) if confidences else None
            }
            interval_data.append(entry)

    df = pd.DataFrame(interval_data)
    df.to_csv(csv_path, index=False)

# ==== VISUAL ====

def extract_video_frames(video_path, out_dir="frames", every_n_sec=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    success = True

    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, count * every_n_sec * 1000)
        success, frame = cap.read()
        if success:
            filename = os.path.join(out_dir, f"frame_{count * every_n_sec:.2f}s.jpg")
            cv2.imwrite(filename, frame)
            count += 1
    cap.release()
    return count

def extract_features(video_path):
    print("Extracting audio...")
    audio_path = extract_audio(video_path)

    print("Extracting audio features...")
    extract_audio_features(audio_path)

    print("Transcribing commentary...")
    transcribe_audio(audio_path)

    print("Extracting visual frames...")
    extract_video_frames(video_path)

extract_features("Olympiakos vs Aek 6-0 full match.mp4")
# extract_features("test.mp4")
