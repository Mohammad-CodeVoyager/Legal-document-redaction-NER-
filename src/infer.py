import argparse
from typing import Any, Dict, List, Tuple

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_overlapping_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Merge overlapping/adjacent spans of the same label.
    Input spans: (start, end, label) with character offsets.
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[2], x[0], x[1]))  # group by label then position

    merged: List[Tuple[int, int, str]] = []
    cur_s, cur_e, cur_lab = spans[0]

    for s, e, lab in spans[1:]:
        if lab == cur_lab and s <= cur_e:  # overlap or touch
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_lab))
            cur_s, cur_e, cur_lab = s, e, lab

    merged.append((cur_s, cur_e, cur_lab))
    return merged


def _labels_to_spans(
    offsets: List[Tuple[int, int]],
    pred_label_ids: List[int],
    id2label: Dict[int, str],
) -> List[Tuple[int, int, str]]:
    """
    Convert token-level predictions to entity spans (char offsets).
    Uses BIO tags: B-XXX / I-XXX / O.
    """
    spans: List[Tuple[int, int, str]] = []
    cur_start = None
    cur_end = None
    cur_type = None

    def close_entity():
        nonlocal cur_start, cur_end, cur_type
        if cur_start is not None and cur_end is not None and cur_type is not None:
            spans.append((cur_start, cur_end, cur_type))
        cur_start, cur_end, cur_type = None, None, None

    for (s, e), pid in zip(offsets, pred_label_ids):
        if s == e:
            continue  # special tokens

        tag = id2label[int(pid)]
        if tag == "O":
            close_entity()
            continue

        # tag like "B-PERSON" or "I-ORG"
        if "-" not in tag:
            close_entity()
            continue

        prefix, ent_type = tag.split("-", 1)

        if prefix == "B":
            close_entity()
            cur_start, cur_end, cur_type = s, e, ent_type
        elif prefix == "I":
            # continue only if same type; otherwise start new entity
            if cur_type == ent_type and cur_start is not None:
                cur_end = e
            else:
                close_entity()
                cur_start, cur_end, cur_type = s, e, ent_type
        else:
            close_entity()

    close_entity()
    return spans


def redact_text(text: str, spans: List[Tuple[int, int, str]], style: str = "type") -> str:
    """
    Replace entity spans in text with placeholders.
    style="type" -> [PERSON], [ORG], etc.
    """
    if not spans:
        return text

    # Sort by start; apply from left to right
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))

    out = []
    last = 0
    for s, e, t in spans_sorted:
        if s < last:
            # overlapping span already handled; skip
            continue
        out.append(text[last:s])
        if style == "type":
            out.append(f"[{t}]")
        else:
            out.append("[REDACTED]")
        last = e
    out.append(text[last:])
    return "".join(out)


@torch.no_grad()
def predict(
    text: str,
    tokenizer,
    model,
    max_length: int,
    stride: int,
    device: str,
) -> List[Tuple[int, int, str]]:
    """
    Run chunked inference and return merged spans (start, end, type).
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
    )

    id2label = model.config.id2label

    all_spans: List[Tuple[int, int, str]] = []

    for input_ids, attn, offsets in zip(enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]):
        input_ids_t = torch.tensor([input_ids], device=device)
        attn_t = torch.tensor([attn], device=device)

        logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()

        spans = _labels_to_spans(offsets, pred_ids, id2label)
        all_spans.extend(spans)

    # Merge duplicates caused by overlapping chunks
    all_spans = _merge_overlapping_spans(all_spans)
    return all_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--model_dir", default=None, help="Defaults to project.output_dir from config.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--redact", action="store_true", help="Print redacted text output.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_dir = args.model_dir or cfg["project"]["output_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    spans = predict(
        text=args.text,
        tokenizer=tokenizer,
        model=model,
        max_length=int(cfg["model"]["max_length"]),
        stride=int(cfg["model"]["stride"]),
        device=device,
    )

    print("\n Predicted entity spans:")
    for s, e, t in spans:
        print(f"{t:10s}  [{s},{e})  '{args.text[s:e]}'")

    if args.redact:
        style = cfg.get("inference", {}).get("placeholder_style", "type")
        redacted = redact_text(args.text, spans, style=style)
        print("\n Redacted text:")
        print(redacted)


if __name__ == "__main__":
    main()