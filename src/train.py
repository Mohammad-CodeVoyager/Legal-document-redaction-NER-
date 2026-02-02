import argparse
import os
from typing import Any, Dict

import yaml
import numpy as np
import evaluate

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from utils import set_seed
from dataloader import prepare_datasets


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_metrics_fn(label_list):
    """
    Uses seqeval to compute token-classification NER metrics.
    """
    seqeval = evaluate.load("seqeval")
    id2label = {i: l for i, l in enumerate(label_list)}

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        true_preds, true_labels = [], []
        for pred_seq, lab_seq in zip(preds, labels):
            cur_p, cur_l = [], []
            for pred_id, lab_id in zip(pred_seq, lab_seq):
                if lab_id == -100:
                    continue
                cur_p.append(id2label[int(pred_id)])
                cur_l.append(id2label[int(lab_id)])
            true_preds.append(cur_p)
            true_labels.append(cur_l)

        m = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": m.get("overall_precision", 0.0),
            "recall": m.get("overall_recall", 0.0),
            "f1": m.get("overall_f1", 0.0),
            "accuracy": m.get("overall_accuracy", 0.0),
        }

    return compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # reproducibility
    set_seed(int(cfg["project"]["seed"]))

    # Load datasets + tokenizer + label maps (from local JSON splits)
    tokenizer, label_list, label2id, id2label, train_ds, val_ds, _ = prepare_datasets(cfg)

    # Build model
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"]["pretrained_name"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Trainer config
    output_dir = cfg["project"]["output_dir"]
    logging_dir = cfg["project"].get("logging_dir", os.path.join(output_dir, "logs"))

    t = cfg["training"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        run_name=cfg["project"].get("run_name", None),

        num_train_epochs=float(t["num_train_epochs"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        warmup_ratio=float(t.get("warmup_ratio", 0.0)),

        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1)),

        evaluation_strategy=str(t["evaluation_strategy"]),
        eval_steps=int(t["eval_steps"]),

        save_strategy=str(t["save_strategy"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t.get("save_total_limit", 2)),

        logging_strategy=str(t["logging_strategy"]),
        logging_steps=int(t["logging_steps"]),

        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        metric_for_best_model=str(t["metric_for_best_model"]),
        greater_is_better=bool(t["greater_is_better"]),

        fp16=bool(t.get("fp16", False)),
        bf16=bool(t.get("bf16", False)),

        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        report_to=t.get("report_to", []),
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_metrics_fn(label_list),
    )

    # Train
    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n Training complete. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
