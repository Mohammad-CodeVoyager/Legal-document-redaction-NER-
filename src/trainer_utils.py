from typing import Any, Dict
from transformers import TrainingArguments


def build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    """
    Builds TrainingArguments from cfg['training'] and cfg['project'].
    Handles transformers naming differences (evaluation_strategy vs eval_strategy).
    """
    t = cfg["training"]
    output_dir = cfg["project"]["output_dir"]
    logging_dir = cfg["project"].get("logging_dir", f"{output_dir}/logs")

    base_kwargs = dict(
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

        eval_steps=int(t["eval_steps"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t.get("save_total_limit", 2)),
        logging_steps=int(t["logging_steps"]),

        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        metric_for_best_model=str(t["metric_for_best_model"]),
        greater_is_better=bool(t["greater_is_better"]),

        fp16=bool(t.get("fp16", False)),
        bf16=bool(t.get("bf16", False)),

        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        report_to=t.get("report_to", []),
    )

    eval_value = str(t["evaluation_strategy"])
    save_value = str(t["save_strategy"])
    log_value = str(t["logging_strategy"])

    # transformers compatibility: evaluation_strategy renamed to eval_strategy in some builds
    try:
        return TrainingArguments(
            **base_kwargs,
            eval_strategy=eval_value,
            save_strategy=save_value,
            logging_strategy=log_value,
        )
    except TypeError:
        return TrainingArguments(
            **base_kwargs,
            eval_strategy=eval_value,
            save_strategy=save_value,
            logging_strategy=log_value,
        )


def build_eval_args(cfg: Dict[str, Any]) -> TrainingArguments:
    """
    Lightweight TrainingArguments for evaluation-only runs.
    Values live in cfg['eval_run'].
    """
    e = cfg.get("eval_run", {})
    bs = int(e.get("per_device_eval_batch_size", cfg["training"]["per_device_eval_batch_size"]))
    out = e.get("output_dir", "outputs/_eval_tmp")

    return TrainingArguments(
        output_dir=out,
        per_device_eval_batch_size=bs,
        report_to=[],
    )