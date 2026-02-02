import os
from typing import Any, Dict, List, Tuple

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from utils import build_bio_labels, normalize_mentions, spans_to_bio_labels_for_offsets


def build_tokenizer_and_labels(cfg: Dict[str, Any]):
    """
    Creates:
      tokenizer, label_list, label2id, id2label
    from config.
    """
    label_list = build_bio_labels(cfg["labels"]["entity_types"])
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_name"], use_fast=True)
    return tokenizer, label_list, label2id, id2label


def load_local_splits(cfg: Dict[str, Any]) -> DatasetDict:
 
    data_cfg = cfg["data"]
    data_dir = os.path.abspath(data_cfg["local_data_dir"])

    train_path = os.path.join(data_dir, data_cfg["train_file"])
    val_path = os.path.join(data_dir, data_cfg["val_file"])
    test_path = os.path.join(data_dir, data_cfg["test_file"])

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset file: {p}")

    # Returns a DatasetDict with keys: train / validation / test
    dsd = load_dataset("json", data_files={"train": train_path, "validation": val_path, "test": test_path})
    return dsd


def tokenize_and_align_labels(ds, cfg: Dict[str, Any], tokenizer, label2id: Dict[str, int]):
    """
    Converts each example to one or more chunks using (max_length, stride),
    and converts entity span annotations -> token BIO label IDs via offset mapping.

    Output columns:
      - input_ids
      - attention_mask
      - labels
    """
    text_field = cfg["data"]["text_field"]
    mentions_field = cfg["data"]["mentions_field"]
    allowed_types = set(cfg["labels"]["entity_types"])

    max_length = int(cfg["model"]["max_length"])
    stride = int(cfg["model"]["stride"])

    def preprocess(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        out_input_ids: List[List[int]] = []
        out_attention: List[List[int]] = []
        out_labels: List[List[int]] = []

        for text, mentions in zip(batch[text_field], batch[mentions_field]):
            norm_mentions = normalize_mentions(mentions)

            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )

            for input_ids, attn, offsets in zip(
                enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]
            ):
                labels = spans_to_bio_labels_for_offsets(
                    offsets=offsets,
                    mentions=norm_mentions,
                    label2id=label2id,
                    allowed_types=allowed_types,
                )
                out_input_ids.append(input_ids)
                out_attention.append(attn)
                out_labels.append(labels)

        return {"input_ids": out_input_ids, "attention_mask": out_attention, "labels": out_labels}

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


def prepare_datasets(cfg: Dict[str, Any]):
    """
    One-call function used by train.py and eval.py.

    Returns:
      tokenizer, label_list, label2id, id2label, train_ds, val_ds, test_ds
    """
    dsd = load_local_splits(cfg)
    tokenizer, label_list, label2id, id2label = build_tokenizer_and_labels(cfg)

    train_ds = tokenize_and_align_labels(dsd["train"], cfg, tokenizer, label2id)
    val_ds = tokenize_and_align_labels(dsd["validation"], cfg, tokenizer, label2id)
    test_ds = tokenize_and_align_labels(dsd["test"], cfg, tokenizer, label2id)

    return tokenizer, label_list, label2id, id2label, train_ds, val_ds, test_ds
