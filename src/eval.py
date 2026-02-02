import argparse
from typing import Any, Dict

import yaml
import numpy as np
import evaluate

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from data_loader import prepare_datasets


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
    ap.add_argument(
        "--model_dir",
        default=None,
        help="Path to a saved model directory. Defaults to project.output_dir from config.",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_dir = args.model_dir or cfg["project"]["output_dir"]

    # Prepare test dataset using the same tokenizer/label space as training
    tokenizer, label_list, _, _, _, _, test_ds = prepare_datasets(cfg)

    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Minimal args needed for Trainer.evaluate()
    dummy_args = TrainingArguments(
        output_dir="outputs/_eval_tmp",
        per_device_eval_batch_size=int(cfg["training"]["per_device_eval_batch_size"]),
        report_to=[],
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=dummy_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_metrics_fn(label_list),
    )

    metrics = trainer.evaluate()
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
