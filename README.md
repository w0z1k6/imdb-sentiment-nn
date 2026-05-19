# IMDB sentiment — TF-IDF + neural network (GitHub Actions → Hugging Face)

Course-style ML project: binary sentiment on IMDB-style reviews using **TF-IDF** features and a small **PyTorch MLP**, trained from `imdb_top_500.csv`, with CI/CD that trains on push and uploads artifacts to the Hugging Face Hub.

## Layout

```text
├── data/                 # place imdb_top_500.csv here (see below)
├── model/                # outputs from train.py (gitignored)
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .github/workflows/train-and-upload.yml
```

## Dataset

Use **`imdb_top_500.csv`** with columns `text`, `label` (and optional `rating`). Put the file at **`data/imdb_top_500.csv`**, or keep a copy in the repo root; `train.py` checks `data/` first, then the root.

## Local usage

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python train.py
python predict.py "This movie was surprisingly good."
```

Environment overrides (optional): `DATA_CSV`, `SEED`, `MAX_FEATURES`, `NGRAM_MAX`, `HIDDEN` (e.g. `256,128`), `DROPOUT`, `LR`, `EPOCHS`.

## GitHub → Hugging Face

1. Create a **model** repo on [Hugging Face](https://huggingface.co/new) (e.g. `your-username/imdb-sentiment-nn`).
2. In the GitHub repo: **Settings → Secrets and variables → Actions**, add:
   - `HF_TOKEN` — Hugging Face access token with write access.
   - `HF_REPO_ID` — full id, e.g. `your-username/imdb-sentiment-nn`.
3. Push to **`main`** or **`master`**. The workflow installs deps, runs `train.py`, then uploads `model/model.pt`, `model/vectorizer.pkl`, `model/config.json`, and `model/metrics.json` to the Hub.

Per assignment constraints: do not manually upload model weights; rely on Actions after the Hub repo exists.

## Deliverables checklist

- [ ] GitHub repository with this structure  
- [ ] Secrets `HF_TOKEN` and `HF_REPO_ID` configured  
- [ ] Successful workflow run; artifacts visible on the Hub  

## License

Educational use. Dataset belongs to its original provider / course materials.
