import argparse
from typing import Any, Dict, List, Tuple

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_overlapping_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[2], x[0], x[1]))
    merged: List[Tuple[int, int, str]] = []
    cur_s, cur_e, cur_lab = spans[0]
    for s, e, lab in spans[1:]:
        if lab == cur_lab and s <= cur_e:
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
            continue
        tag = id2label[int(pid)]
        if tag == "O" or "-" not in tag:
            close_entity()
            continue

        prefix, ent_type = tag.split("-", 1)
        if prefix == "B":
            close_entity()
            cur_start, cur_end, cur_type = s, e, ent_type
        elif prefix == "I":
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
    if not spans:
        return text
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    out = []
    last = 0
    for s, e, t in spans_sorted:
        if s < last:
            continue
        out.append(text[last:s])
        out.append(f"[{t}]" if style == "type" else "[REDACTED]")
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
    score_threshold: float = 0.0,
) -> List[Tuple[int, int, str]]:
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
    o_id = model.config.label2id.get("O", 0)

    all_spans: List[Tuple[int, int, str]] = []

    for input_ids, attn, offsets in zip(enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]):
        input_ids_t = torch.tensor([input_ids], device=device)
        attn_t = torch.tensor([attn], device=device)

        logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits[0]
        probs = torch.softmax(logits, dim=-1)
        conf, pred_ids = torch.max(probs, dim=-1)

        pred_ids = pred_ids.tolist()
        conf = conf.tolist()

        filtered = [(pid if c >= score_threshold else o_id) for pid, c in zip(pred_ids, conf)]
        spans = _labels_to_spans(offsets, filtered, id2label)
        all_spans.extend(spans)

    return _merge_overlapping_spans(all_spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--text", required=True)
    ap.add_argument("--redact", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model_dir = args.model_dir or cfg["project"]["output_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    score_threshold = float(cfg.get("inference", {}).get("score_threshold", 0.0))
    spans = predict(
        text=args.text,
        tokenizer=tokenizer,
        model=model,
        max_length=int(cfg["model"]["max_length"]),
        stride=int(cfg["model"]["stride"]),
        device=device,
        score_threshold=score_threshold,
    )

    print("\n Predicted entity spans:")
    for s, e, t in spans:
        print(f"{t:10s}  [{s},{e})  '{args.text[s:e]}'")

    if args.redact:
        style = cfg.get("inference", {}).get("placeholder_style", "type")
        print("\n Redacted text:")
        print(redact_text(args.text, spans, style=style))


if __name__ == "__main__":
    main()