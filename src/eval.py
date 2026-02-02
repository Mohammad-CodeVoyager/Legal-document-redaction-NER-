import argparse
import json
import os
from typing import Any, Dict

import yaml
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

from dataloader import prepare_datasets
from trainer_utils import build_eval_args
from utils import build_seqeval_metrics


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--model_dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_dir = args.model_dir or cfg["project"]["output_dir"]

    tokenizer, label_list, _, _, _, _, test_ds = prepare_datasets(cfg)

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    eval_args = build_eval_args(cfg)
    collator = DataCollatorForTokenClassification(tokenizer)

    metrics_cfg = cfg.get("metrics", {})
    compute_metrics = build_seqeval_metrics(label_list) if metrics_cfg.get("enabled", True) else None

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    out_path = os.path.join(cfg["project"]["output_dir"], "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved test metrics to: {out_path}")

    print("\n Test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()