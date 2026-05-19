# ML Project: GitHub CI/CD → Hugging Face

> [!INFO]
> Your final delivery should resemble:
>
> * [https://github.com/EmporioSabo/california-housing-predictor](https://github.com/EmporioSabo/california-housing-predictor)
> * [https://huggingface.co/EmporioSabo/california-housing-predictor/tree/main](https://huggingface.co/EmporioSabo/california-housing-predictor/tree/main)
>
> Once completed, post both URLs in the WeChat group.

---

### Objective

Build and deploy a complete Machine Learning project pipeline:

1. Design a simple but meaningful ML project.
2. Upload the full codebase to GitHub.
3. Configure GitHub Actions CI/CD to automatically:

   * Train your model
   * Save artifacts
   * Upload everything to Hugging Face Hub
4. After creating your Hugging Face model repository once, all future updates must happen automatically through GitHub Actions — no manual uploads.

---

### Your Specific Task

Follow the structure described in:

`practice-huggingface-with-github-ci-cd.md`

But instead of a generic ML model, your project should focus on:

#### Neural Network Sentiment Analysis Project

Choose one of the following datasets:

* `imdb_top_500.csv`
* `imdb_balanced_10k.csv`

#### Recommended Modeling Options

##### Option A — Simpler (Recommended)

**BoW / TF-IDF + Feedforward Neural Network**

* Text preprocessing
* Tokenization
* TF-IDF or Bag-of-Words vectorization
* Dense neural network classifier

##### Option B — More Advanced

**tiny_glove.json Embeddings + Embedding Layer / MLP**

* Token embedding using tiny GloVe
* Sequence or pooled representation
* Neural network sentiment classifier

---

### Deliverables

Your repository should include:

#### GitHub Repository

A clean, professional ML engineering project structure:

```bash
your-project/
│
├── data/
├── model/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── train-and-upload.yml
```

---

### Required CI/CD Flow

#### On Push to `main` or `master`:

GitHub Actions should automatically:

##### Step 1 — Setup

* Checkout repository
* Install Python
* Install dependencies

##### Step 2 — Train

* Run `train.py`
* Save:

  * model weights (`model.pt`)
  * tokenizer/vectorizer (`vectorizer.pkl`)
  * config (`config.json`)
  * metrics (`metrics.json`)

##### Step 3 — Deploy

* Authenticate using `HF_TOKEN`
* Create Hugging Face repo if needed
* Upload artifacts automatically

---

### Important Constraint

#### You should NOT:

* Manually upload model files to Hugging Face
* Re-train locally just for deployment
* Edit Hugging Face repo after initial creation

#### You SHOULD:

* Push code to GitHub
* Let GitHub Actions handle everything

---

### Example Inspiration

#### GitHub Repo Structure

* Space Mining GitHub Repo:
  [https://github.com/reveurmichael/space_mining/tree/main](https://github.com/reveurmichael/space_mining/tree/main)

#### Hugging Face Hub Example

* Space Mining HF Hub:
  [https://huggingface.co/LUNDECHEN/space-mining-ppo/tree/main](https://huggingface.co/LUNDECHEN/space-mining-ppo/tree/main)

#### CI/CD Workflow Reference

* GitHub Actions Example:
  [https://github.com/reveurmichael/space_mining/blob/main/.github/workflows/train-long-wandb-hf.yml](https://github.com/reveurmichael/space_mining/blob/main/.github/workflows/train-long-wandb-hf.yml)

---

### Suggested Project Ideas

#### Example Repository Names

* `imdb-sentiment-nn`
* `imdb-review-classifier`
* `sentiment-analysis-ci-cd`
* `imdb-tfidf-neural-net`

---

### Hugging Face Secret Setup

In your GitHub repository:

#### Settings → Secrets and Variables → Actions

Add:

```bash
HF_TOKEN = your_huggingface_access_token
```

---

### Workflow Example (Customize It)

```yaml
name: Train and Upload to Hugging Face

on:
  push:
    branches: [main, master]
  workflow_dispatch:

jobs:
  train-and-upload:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python train.py

      - name: Upload to Hugging Face Hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python - <<'EOF'
          import os
          from huggingface_hub import HfApi

          token = os.environ["HF_TOKEN"]
          repo_id = "YOUR_HF_REPO"
          api = HfApi()

          # Create repo if it doesn't exist
          api.create_repo(repo_id, token=token, exist_ok=True)

          # Upload all model artifacts
          for filename in [FILES_TO_UPLOAD_TO_HF]:
              api.upload_file(
                  path_or_fileobj=filename,
                  path_in_repo=filename,
                  repo_id=repo_id,
                  token=token,
              )
              print(f"Uploaded {filename}")

          print(f"Model uploaded to https://huggingface.co/{repo_id}")
          EOF
```


---

### Final Submission

After everything works:

#### Send to WeChat Group:

* GitHub Repository URL
* Hugging Face Repository URL
