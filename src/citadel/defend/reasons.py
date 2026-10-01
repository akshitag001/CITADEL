"""Reason codes and customer messages.

Attribution method (DECISIONS.md D-008): RULE CARDS, not SHAP. Every reason code is a fixed, readable
predicate over features. For an alerting row we collect the codes whose predicate holds, put L0 rule
codes first, then order the rest by the permutation importance of the code's feature family in the L1
model (measured on the stats slice). This is stable across retrains, auditable by a regulator, and a
code is only ever shown when its plain-language statement is literally true for that payment. When no
card holds (rare), the generic R13 is used and counted as a fallback in defend_reason_code_usage.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import CONFIG_DIR
from .l0 import _eval, _parse

MAX_REASONS = 3


def load_reason_codes(path: Path | None = None) -> dict:
    raw = yaml.safe_load(Path(path or CONFIG_DIR / "reason_codes.yaml").read_text(encoding="utf-8"))
    codes = raw["codes"]
    for code, spec in codes.items():
        if spec.get("predicate"):
            spec["_tree"], _ = _parse(code, spec["predicate"])
    return codes


def load_messages(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "messages.yaml").read_text(encoding="utf-8"))


def predicate_matrix(codes: dict, X: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for code, spec in codes.items():
        if spec.get("_tree") is not None:
            cols[code] = np.asarray(_eval(spec["_tree"], X), dtype=bool)
    return pd.DataFrame(cols, index=X.index)


def assign_reasons(codes: dict, X: pd.DataFrame, action_level: np.ndarray, l0_codes: list[list[str]],
                   abstained: np.ndarray, family_weight: dict[str, float]) -> tuple[list[list[str]], np.ndarray]:
    """1-3 codes per alerting row (empty for A0). Returns (codes per row, fallback flag per row)."""
    P = predicate_matrix(codes, X)
    order = sorted(P.columns, key=lambda c: -family_weight.get(codes[c]["family"], 0.0))
    Pv = P[order].to_numpy()
    out: list[list[str]] = []
    fallback = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if action_level[i] == 0:
            out.append([])
            continue
        chosen: list[str] = []
        for c in l0_codes[i]:
            if c not in chosen:
                chosen.append(c)
        if abstained[i] and "R15" not in chosen:
            chosen.append("R15")
        for j in np.flatnonzero(Pv[i]):
            if len(chosen) >= MAX_REASONS:
                break
            if order[j] not in chosen:
                chosen.append(order[j])
        if not chosen:
            chosen = ["R13"]
            fallback[i] = True
        # keep R14 only when an L0 rule actually fired; trim to the closed maximum
        out.append(chosen[:MAX_REASONS])
    return out, fallback


def customer_message(messages: dict, action: str, reasons: list[str], txn_type: str, lang: str,
                     amount: float | None = None) -> dict:
    """The exact payload a UPI app renders. Never contains a score."""
    a = messages["actions"][action]
    fmt = {"cooling_s": a.get("cooling_s", 0), "sla_min": a.get("sla_min", 0),
           "amount": f"{amount:,.0f}" if amount is not None else ""}
    if action == "A0":
        return {"action": "A0", "title": "", "body": "", "reasons": [], "screen": "", "buttons": []}
    return {
        "action": action,
        "title": a["title"][lang],
        "body": a["body"][lang].format(**fmt),
        "reasons": [messages["reasons"][c][lang] for c in reasons],
        "screen": messages["txn_type_screens"][txn_type][lang].format(**fmt),
        "buttons": [{"id": b, "label": messages["buttons"][b][lang]} for b in a["buttons"]],
        "cooling_off_s": int(a.get("cooling_s", 0)),
        "review_sla_min": int(a.get("sla_min", 0)),
    }
