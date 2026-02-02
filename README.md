# Legal Document Redaction & NER (DistilBERT)

Fine-tune a transformer for Named Entity Recognition (NER) to detect personal information in legal text and support redaction for compliance.

## Dataset
We are using the **Text Anonymization Benchmark (TAB)** which contains **1,268 ECHR court cases** annotated for anonymization.

Source: https://huggingface.co/datasets/ildpil/text-anonymization-benchmark

> Note: This repository may include dataset files under `data/` (if you keep a local copy).  
> The training pipeline can also load TAB directly from Hugging Face Datasets.

---

## Repository Structure
- `src/` — training, evaluation, inference code
- `configs/` — YAML configuration files
- `docs/` — documentation (dataset notes, pipeline explanation, experiments log)
- `outputs/` — training artifacts (checkpoints/model files) **ignored by git**
- `data/` — optional dataset storage (if you keep the dataset in the repo)

---

## Setup

### Local (Mac / Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
```

### Google Colab
```bash
pip install -r Requirements.txt
```

---

## Train
```bash
python src/train.py --config configs/training_config.yaml
```

---

## Evaluate
```bash
python src/eval.py --config configs/training_config.yaml
```

---

## Inference
```bash
python src/infer.py --config configs/training_config.yaml --text "John lives in Toronto."
```

---

## Outputs / Artifacts
Training outputs are saved under:

- `outputs/` (e.g., `outputs/distilbert_tab/`)

This folder is intentionally **not committed** to GitHub because it may contain large model weights and checkpoints.

---

## Model Publishing (Recommended)
For deployment (e.g., later Streamlit inference), publish the final model to **Hugging Face Hub**:
- This avoids bloating the Git repo with large binary model files.
- Your app can load the model using `from_pretrained("<username>/<model_name>")`.

---

## Notes
- TAB documents are long; training typically uses `max_length` + `stride` chunking.
- If you change entity label policy (e.g., which entity types to include), update `configs/training_config.yaml` and retrain.

---

## License / Credits
- Dataset: Text Anonymization Benchmark (TAB) — see dataset card on Hugging Face for citation and license.
