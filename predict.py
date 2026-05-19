"""
Load trained TF-IDF + MLP artifacts and predict sentiment for given text.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent / "model"


def clean_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.0)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_artifacts(device: torch.device):
    path = MODEL_DIR / "model.pt"
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    hidden = tuple(ckpt.get("hidden", (256, 128)))
    dropout = float(ckpt.get("dropout", 0.25))
    n_in = int(ckpt["n_features"])
    model = MLP(n_in, hidden, dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vectorizer


def predict_one(text: str, model: nn.Module, vectorizer, device: torch.device) -> tuple[float, int]:
    x = vectorizer.transform([clean_text(text)]).toarray().astype(np.float32)
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logit = model(xt).cpu().numpy().item()
    prob_pos = float(1 / (1 + np.exp(-logit)))
    label = 1 if logit >= 0 else 0
    return prob_pos, label


def main() -> None:
    parser = argparse.ArgumentParser(description="IMDB sentiment (TF-IDF + MLP)")
    parser.add_argument("text", nargs="?", default=None, help="Review text (optional if using stdin)")
    args = parser.parse_args()

    if not (MODEL_DIR / "model.pt").is_file() or not (MODEL_DIR / "vectorizer.pkl").is_file():
        print("Run train.py first to create model/ artifacts.", file=sys.stderr)
        sys.exit(1)

    text = args.text
    if text is None:
        text = sys.stdin.read()
    if not text.strip():
        print("No text provided.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vectorizer = load_artifacts(device)
    prob, label = predict_one(text, model, vectorizer, device)

    if (MODEL_DIR / "config.json").is_file():
        with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    out = {
        "label": label,
        "label_name": "positive" if label == 1 else "negative",
        "prob_positive": prob,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
