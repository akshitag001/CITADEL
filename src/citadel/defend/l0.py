"""Layer 0 deterministic guards.

No model dependency: this module imports nothing from the model stack, so "model down" can never mean
"everything approved". Rules can only escalate friction or place a hold; there is no decline.
Conditions are parsed with ``ast`` and evaluated by a tiny vectorised interpreter (no ``eval``), and
the loader rejects any name that is not a schema column or a registered feature.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config import CONFIG_DIR
from ..features.registry import FEATURES
from ..schema import COLUMNS, TRUTH_ONLY

ACTIONS = ("A0", "A1", "A2", "A3")
LEVEL = {a: i for i, a in enumerate(ACTIONS)}
RAW_FOR_L0 = ["txn_type", "request_source_known_contact", "collect_request_age_s", "amount_inr",
              "user_id", "payee_id"]


class RuleError(ValueError):
    pass


@dataclass
class Rule:
    id: str
    title: str
    condition: str
    min_action: str
    reason_codes: list[str]
    rationale: str
    log_only: bool = False
    tree: Any = field(default=None, repr=False)
    names: set[str] = field(default_factory=set)


_ALLOWED_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare, ast.Name,
                  ast.Load, ast.Constant, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _parse(rule_id: str, text: str) -> tuple[Any, set[str]]:
    try:
        tree = ast.parse(" ".join(text.split()), mode="eval")
    except SyntaxError as e:
        raise RuleError(f"{rule_id}: cannot parse condition: {e}") from e
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleError(f"{rule_id}: '{type(node).__name__}' is not allowed in a rule condition")
        if isinstance(node, ast.Name):
            names.add(node.id)
    truth = {n for n in names if n in COLUMNS and COLUMNS[n].role == TRUTH_ONLY}
    if truth:
        raise RuleError(f"{rule_id}: TRUTH_ONLY columns {sorted(truth)} cannot appear in a rule")
    known = set(FEATURES) | {c for c, s in COLUMNS.items() if s.role != TRUTH_ONLY}
    unknown = names - known
    if unknown:
        raise RuleError(f"{rule_id}: unknown column(s) {sorted(unknown)}")
    return tree, names


def load_rules(path: str | Path | None = None, raw: dict | None = None) -> list[Rule]:
    if raw is None:
        raw = yaml.safe_load(Path(path or CONFIG_DIR / "l0_rules.yaml").read_text(encoding="utf-8"))
    rules = []
    for r in raw["rules"]:
        if r["min_action"] not in LEVEL:
            raise RuleError(f"{r['id']}: min_action must be one of {ACTIONS}")
        if r["min_action"] == "decline":
            raise RuleError(f"{r['id']}: L0 never declines")
        tree, names = _parse(r["id"], r["condition"])
        rules.append(Rule(id=r["id"], title=r["title"], condition=" ".join(r["condition"].split()),
                          min_action=r["min_action"], reason_codes=list(r["reason_codes"]),
                          rationale=" ".join(r["rationale"].split()), log_only=bool(r.get("log_only", False)),
                          tree=tree, names=names))
    return rules


def _eval(node: Any, frame: pd.DataFrame) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, frame)
    if isinstance(node, ast.BoolOp):
        vals = [np.asarray(_eval(v, frame), dtype=bool) for v in node.values]
        return np.logical_and.reduce(vals) if isinstance(node.op, ast.And) else np.logical_or.reduce(vals)
    if isinstance(node, ast.UnaryOp):
        return ~np.asarray(_eval(node.operand, frame), dtype=bool)
    if isinstance(node, ast.Name):
        return frame[node.id].to_numpy()
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        left = _eval(node.left, frame)
        out = None
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, frame)
            with np.errstate(invalid="ignore"):
                if isinstance(op, ast.Eq):
                    res = np.asarray(left == right)
                elif isinstance(op, ast.NotEq):
                    res = np.asarray(left != right)
                elif isinstance(op, ast.Lt):
                    res = np.asarray(left < right)
                elif isinstance(op, ast.LtE):
                    res = np.asarray(left <= right)
                elif isinstance(op, ast.Gt):
                    res = np.asarray(left > right)
                else:
                    res = np.asarray(left >= right)
            res = res.astype(bool)
            res = res & _known(left) & _known(right)  # NaN never satisfies any comparison, not even !=
            out = res if out is None else out & res
            left = right
        return out
    raise RuleError(f"unsupported node {type(node).__name__}")


def _known(v: Any) -> Any:
    a = np.asarray(v)
    if a.dtype.kind == "f":
        return ~np.isnan(a)
    return True


def l0_frame(tx: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    raw = tx[[c for c in RAW_FOR_L0 if c in tx.columns]].reset_index(drop=True)
    raw["txn_type"] = raw["txn_type"].astype(str)
    return pd.concat([X.reset_index(drop=True), raw], axis=1)


@dataclass
class L0Result:
    fired: pd.DataFrame          # bool per rule
    min_level: np.ndarray        # 0..3, from acting rules
    log_flags: np.ndarray        # bool, log-only rules fired

    def fired_ids(self, i: int) -> list[str]:
        row = self.fired.iloc[i]
        return [c for c in self.fired.columns if row[c]]


def evaluate(rules: list[Rule], frame: pd.DataFrame) -> L0Result:
    n = len(frame)
    fired = {}
    level = np.zeros(n, dtype=np.int8)
    log_flags = np.zeros(n, dtype=bool)
    for rule in rules:
        hit = np.asarray(_eval(rule.tree, frame), dtype=bool)
        if hit.shape != (n,):
            hit = np.broadcast_to(hit, (n,)).copy()
        fired[rule.id] = hit
        if rule.log_only:
            log_flags |= hit
        else:
            level = np.maximum(level, np.where(hit, LEVEL[rule.min_action], 0)).astype(np.int8)
    return L0Result(pd.DataFrame(fired), level, log_flags)


def rules_by_id(rules: list[Rule]) -> dict[str, Rule]:
    return {r.id: r for r in rules}
