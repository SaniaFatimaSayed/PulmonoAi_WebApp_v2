import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn


# =========================
# CONFIG
# =========================
CHECKPOINT_PATH = Path("bigru/segmenter.pt")
DURATION_PRIORS_PATH = Path("bigru/duration_priors.json")

DEVICE = "cpu"

SAMPLE_RATE = 16000
FRAME_MS = 25.0
HOP_MS = 10.0
HIDDEN = 96

ID_TO_LABEL = {
    0: "breath",
    1: "speech"
}


class BiGRUSegmenter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)


def extract_bigru_features(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if y is None or len(y) == 0:
        raise RuntimeError(f"Audio file is empty: {audio_path}")

    if len(y) < SAMPLE_RATE * 2:
        raise RuntimeError("Audio is too short. Please upload at least 20 seconds of recording.")

    n_fft = max(256, int(SAMPLE_RATE * FRAME_MS / 1000.0))
    hop = max(1, int(SAMPLE_RATE * HOP_MS / 1000.0))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=40,
        fmin=50,
        fmax=SAMPLE_RATE // 2,
    )

    mel_db = librosa.power_to_db(mel + 1e-10)

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    n_frames = mfcc.shape[1]

    if n_frames < 3:
        d1 = np.zeros_like(mfcc)
        d2 = np.zeros_like(mfcc)
    else:
        delta_width = min(9, n_frames)

        # width must be odd
        if delta_width % 2 == 0:
            delta_width -= 1

        delta_width = max(3, delta_width)

        d1 = librosa.feature.delta(mfcc, width=delta_width)
        d2 = librosa.feature.delta(mfcc, order=2, width=delta_width)

    feats = np.concatenate(
        [mfcc, d1, d2, mel_db[:20]],
        axis=0
    ).T.astype(np.float32)

    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6

    return (feats - mu) / std


def load_duration_priors(path: Path) -> Dict[int, Dict[str, float]]:
    with path.open("r") as f:
        raw = json.load(f)

    return {
        int(k): v
        for k, v in raw.items()
    }


def duration_logprob(length_f: int, mu: float, sigma: float) -> float:
    if length_f <= 0:
        return -1e9

    x = math.log(length_f)

    return (
        -0.5 * ((x - mu) / sigma) ** 2
        - math.log(length_f * sigma + 1e-8)
    )


def segmental_viterbi(
    log_probs: np.ndarray,
    dur_priors: Dict[int, Dict[str, float]],
    log_trans: np.ndarray
) -> np.ndarray:

    t_max, n_states = log_probs.shape

    max_d = int(
        max(
            dur_priors[0]["p99"],
            dur_priors[1]["p99"]
        ) + 5
    )

    max_d = max(5, min(max_d, t_max))

    csum = np.cumsum(log_probs, axis=0)

    dp = np.full((t_max, n_states), -1e18, dtype=np.float64)
    bp_prev_t = np.full((t_max, n_states), -1, dtype=np.int32)
    bp_prev_s = np.full((t_max, n_states), -1, dtype=np.int32)

    for t in range(t_max):
        dmax = min(max_d, t + 1)

        for s in range(n_states):
            mu = dur_priors[s]["mu"]
            sigma = dur_priors[s]["sigma"]

            best = -1e18
            best_prev_t = -1
            best_prev_s = -1

            for d in range(1, dmax + 1):
                st = t - d + 1

                emit = csum[t, s] - (
                    csum[st - 1, s] if st > 0 else 0.0
                )

                score = emit + duration_logprob(d, mu, sigma)

                if st == 0:
                    if score > best:
                        best = score
                        best_prev_t = -1
                        best_prev_s = -1

                else:
                    for ps in range(n_states):
                        if ps == s:
                            continue

                        cand = dp[st - 1, ps] + log_trans[ps, s] + score

                        if cand > best:
                            best = cand
                            best_prev_t = st - 1
                            best_prev_s = ps

            dp[t, s] = best
            bp_prev_t[t, s] = best_prev_t
            bp_prev_s[t, s] = best_prev_s

    labels = np.full((t_max,), -1, dtype=np.int64)

    s = int(np.argmax(dp[t_max - 1]))
    t = t_max - 1

    while t >= 0:
        pt = bp_prev_t[t, s]
        ps = bp_prev_s[t, s]

        st = 0 if pt < 0 else pt + 1
        labels[st:t + 1] = s

        t = pt
        s = ps if ps >= 0 else 0

    return labels


def collapse_segments(labels: np.ndarray, hop_sec: float) -> List[dict]:
    segments = []

    if len(labels) == 0:
        return segments

    cur = int(labels[0])
    start = 0

    for i in range(1, len(labels)):
        if int(labels[i]) != cur:
            start_sec = start * hop_sec
            end_sec = i * hop_sec

            segments.append({
                "segment_type": ID_TO_LABEL[cur],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec
            })

            start = i
            cur = int(labels[i])

    start_sec = start * hop_sec
    end_sec = len(labels) * hop_sec

    segments.append({
        "segment_type": ID_TO_LABEL[cur],
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": end_sec - start_sec
    })

    return segments


def run_bigru_segmentation(audio_path: str) -> List[dict]:
    features = extract_bigru_features(audio_path)

    in_dim = features.shape[1]

    model = BiGRUSegmenter(
        in_dim=in_dim,
        hidden=HIDDEN
    )

    device = torch.device(DEVICE)

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dur_priors = load_duration_priors(DURATION_PRIORS_PATH)

    trans = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90]
        ],
        dtype=np.float64
    )

    log_trans = np.log(trans + 1e-12)

    with torch.no_grad():
        x = torch.tensor(
            features,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        logits = model(x)[0]

        probs = torch.softmax(
            logits,
            dim=-1
        ).cpu().numpy()

    labels = segmental_viterbi(
        np.log(probs + 1e-12),
        dur_priors,
        log_trans
    )

    hop_sec = HOP_MS / 1000.0

    segments = collapse_segments(
        labels,
        hop_sec
    )

    return segments