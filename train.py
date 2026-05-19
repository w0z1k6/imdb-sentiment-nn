"""
Train a TF-IDF + feedforward neural network for IMDB review sentiment (binary label).
Writes model/model.pt, model/vectorizer.pkl, model/config.json, model/metrics.json.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = Path(__file__).resolve().parent / "model"


def _find_data_csv() -> Path:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "data" / "imdb_top_500.csv",
        root / "imdb_top_500.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find imdb_top_500.csv. Place it in data/ or project root."
    )


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    set_seed(int(os.environ.get("SEED", "42")))

    csv_path = Path(os.environ.get("DATA_CSV", _find_data_csv()))
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: text, label")

    texts = [clean_text(t) for t in df["text"].tolist()]
    y = df["label"].astype(int).values

    max_features = int(os.environ.get("MAX_FEATURES", "4000"))
    ngram_max = int(os.environ.get("NGRAM_MAX", "2"))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts).astype(np.float32)
    n_features = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_t = torch.from_numpy(X_train.toarray())
    X_val_t = torch.from_numpy(X_val.toarray())
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    batch_size = min(64, len(y_train))
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    hidden = tuple(
        int(x)
        for x in os.environ.get("HIDDEN", "256,128").split(",")
        if x.strip()
    ) or (256, 128)
    dropout = float(os.environ.get("DROPOUT", "0.25"))
    lr = float(os.environ.get("LR", "1e-3"))
    epochs = int(os.environ.get("EPOCHS", "80"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_features, hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(y_train)

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t.to(device)).cpu().numpy()
            v_probs = 1 / (1 + np.exp(-v_logits))
            v_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    torch.from_numpy(v_logits),
                    y_val_t,
                    reduction="mean",
                ).item()
            )
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={v_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tr_logits = model(X_train_t.to(device)).cpu().numpy()
        va_logits = model(X_val_t.to(device)).cpu().numpy()
    tr_pred = (tr_logits >= 0).astype(int)
    va_pred = (va_logits >= 0).astype(int)
    tr_acc = float(accuracy_score(y_train, tr_pred))
    va_acc = float(accuracy_score(y_val, va_pred))
    try:
        va_auc = float(roc_auc_score(y_val, va_logits))
    except ValueError:
        va_auc = float("nan")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden": hidden,
            "dropout": dropout,
        },
        MODEL_DIR / "model.pt",
    )
    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    config = {
        "data_csv": str(csv_path.as_posix()),
        "max_features": max_features,
        "ngram_range": [1, ngram_max],
        "hidden": list(hidden),
        "dropout": dropout,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": int(os.environ.get("SEED", "42")),
        "framework": "pytorch",
        "vectorizer": "sklearn_tfidf",
    }
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "train_accuracy": tr_acc,
        "val_accuracy": va_acc,
        "val_roc_auc": va_auc,
        "best_val_loss": best_val,
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", MODEL_DIR / "model.pt")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
