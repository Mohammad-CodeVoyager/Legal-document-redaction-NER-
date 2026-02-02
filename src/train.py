import argparse
from typing import Any, Dict

import yaml
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    EarlyStoppingCallback,
)

from utils import set_seed, build_seqeval_metrics
from dataloader import prepare_datasets
from trainer_utils import build_training_args


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg["project"]["seed"]))

    tokenizer, label_list, label2id, id2label, train_ds, val_ds, _ = prepare_datasets(cfg)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"]["pretrained_name"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = build_training_args(cfg)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    metrics_cfg = cfg.get("metrics", {})
    compute_metrics = build_seqeval_metrics(label_list) if metrics_cfg.get("enabled", True) else None

    callbacks = []
    es_cfg = cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(es_cfg.get("patience", 2)),
                early_stopping_threshold=float(es_cfg.get("threshold", 0.0)),
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    out_dir = cfg["project"]["output_dir"]
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n Training complete. Model saved to: {out_dir}")


if __name__ == "__main__":
    main()