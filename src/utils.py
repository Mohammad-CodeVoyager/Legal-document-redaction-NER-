import random
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bio_labels(entity_types: List[str]) -> List[str]:
    """
    Builds label list like: ["O", "B-PERSON", "I-PERSON", "B-ORG", "I-ORG", ...]
    """
    labels = ["O"]
    for t in entity_types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


def normalize_mentions(mentions: Any) -> List[Dict[str, Any]]:
    """
      Expected mention keys:
      start_offset, end_offset, entity_type
    Returns:
      [{"start": int, "end": int, "type": str}, ...]
    """
   
    if not isinstance(mentions, list):
        return []

    out = []
    for m in mentions:
        if not isinstance(m, dict):
            continue
        s = m.get("start_offset")
        e = m.get("end_offset")
        t = m.get("entity_type")
        if s is None or e is None or t is None:
            continue
        try:
            s = int(s)
            e = int(e)
        except Exception:
            continue
        if e <= s:
            continue
        out.append({"start": s, "end": e, "type": str(t)})
    # sort helps stable labeling
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def _token_overlaps_span(tok_s: int, tok_e: int, span_s: int, span_e: int) -> int:
    """
    Returns overlap length between [tok_s, tok_e) and [span_s, span_e)
    """
    left = max(tok_s, span_s)
    right = min(tok_e, span_e)
    return max(0, right - left)


def spans_to_bio_labels_for_offsets(
    offsets: List[Tuple[int, int]],
    mentions: List[Dict[str, Any]],
    label2id: Dict[str, int],
    allowed_entity_types: Optional[set] = None,
) -> List[int]:
    """
    Convert character-level spans to BIO token labels using tokenizer offset mapping.

    offsets: list of (start_char, end_char) per token (special tokens usually (0,0))
    mentions: normalized list: [{"start": int, "end": int, "type": str}, ...]
    allowed_entity_types: if provided, ignore mentions whose type is not in the set
    """
    labels = []
    prev_ent: Optional[str] = None
    prev_in_span: bool = False

    for (tok_s, tok_e) in offsets:
        # Special tokens or padding -> ignore in loss
        if tok_s == tok_e:
            labels.append(-100)
            prev_ent = None
            prev_in_span = False
            continue

        best = None
        best_overlap = 0

        for m in mentions:
            m_type = m["type"]
            if allowed_entity_types is not None and m_type not in allowed_entity_types:
                continue
            ov = _token_overlaps_span(tok_s, tok_e, m["start"], m["end"])
            if ov > best_overlap:
                best_overlap = ov
                best = m

        if best is None or best_overlap == 0:
            labels.append(label2id["O"])
            prev_ent = None
            prev_in_span = False
            continue

        ent = best["type"]

        # BIO logic:
        # - B-ENT if we weren't in same entity on previous token
        # - I-ENT if we were already inside same entity on previous token
        if prev_in_span and prev_ent == ent:
            lab = f"I-{ent}"
        else:
            lab = f"B-{ent}"

        labels.append(label2id.get(lab, label2id["O"]))
        prev_ent = ent
        prev_in_span = True

    return labels
