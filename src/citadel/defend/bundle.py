"""ONE atomic serving bundle: models, calibrators, fusion, ladder, L0 cards, reason cards, messages,
feature order and run provenance, in a single file. ``decide`` is the whole decision path; the API, the
evaluation and the replay bundle all call it, so there is exactly one implementation of a decision.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import l0
from .fusion import Fusion
from .ladder import ACTIONS, BANDS, TXN_TREATMENT, Ladder
from .model import Calibrator, IFModel, L1Model, L2Model
from .reasons import assign_reasons, customer_message

BUNDLE_VERSION = 1


@dataclass
class Bundle:
    l1: L1Model
    l2: L2Model
    iforest: IFModel | None
    calibrators: dict[str, Calibrator]
    fusion: Fusion
    final_calibrator: Calibrator
    ladder: Ladder
    abstain: dict[str, float]
    rank_refs: dict[str, np.ndarray]
    family_weight: dict[str, float]
    features: list[str]
    l0_yaml: str
    reasons_yaml: str
    messages: dict
    provenance: dict
    version: int = BUNDLE_VERSION
    _rules: list = field(default=None, repr=False)
    _codes: dict = field(default=None, repr=False)

    # --- lazily rebuilt (AST objects are not stored) -------------------------------------------------
    @property
    def rules(self) -> list[l0.Rule]:
        if self._rules is None:
            self._rules = l0.load_rules(raw=yaml.safe_load(self.l0_yaml))
        return self._rules

    @property
    def codes(self) -> dict:
        if self._codes is None:
            self._codes = _codes_from_text(self.reasons_yaml)
        return self._codes

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_rules"], d["_codes"] = None, None
        return d

    # --- scoring ------------------------------------------------------------------------------------
    def channel_scores(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        cal = {"L1": self.calibrators["L1"](self.l1.raw(X)), "L2": self.calibrators["L2"](self.l2.raw(X))}
        if self.iforest is not None and "IF" in self.calibrators:
            cal["IF"] = self.calibrators["IF"](self.iforest.raw(X))
        return cal

    def decide(self, X: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        """``X`` feature matrix (bundle feature order), ``raw`` schema rows aligned with X."""
        X = X.reset_index(drop=True)
        n = len(X)
        cal = self.channel_scores(X)
        fused = self.fusion({k: v for k, v in cal.items() if k in self.fusion.channels or k == "L1"})
        risk = self.final_calibrator(fused)
        score_level = self.ladder.level(risk)
        # abstention: high band where the supervised and payee channels disagree -> a human, priced
        r1 = np.searchsorted(self.rank_refs["L1"], cal["L1"], side="right") / len(self.rank_refs["L1"])
        r2 = np.searchsorted(self.rank_refs["L2"], cal["L2"], side="right") / len(self.rank_refs["L2"])
        disagree = np.abs(r1 - r2)
        abstained = ((score_level == 2) & (disagree >= self.abstain["disagreement"])
                     & (risk >= self.abstain["min_risk"]))
        level = np.where(abstained, 3, score_level).astype(np.int8)
        l0r = l0.evaluate(self.rules, l0.l0_frame(raw.reset_index(drop=True), X))
        level = np.maximum(level, l0r.min_level)
        rb = l0.rules_by_id(self.rules)
        l0_ids = [[rid for rid in l0r.fired.columns if l0r.fired[rid].iat[i] and not rb[rid].log_only]
                  for i in range(n)]
        l0_codes = [[c for rid in ids for c in rb[rid].reason_codes] for ids in l0_ids]
        reasons, fallback = assign_reasons(self.codes, X, level, l0_codes, abstained, self.family_weight)
        actions = np.array(ACTIONS)[level]
        ttype = raw["txn_type"].astype(str).to_numpy()
        return pd.DataFrame({
            "p_L1": cal["L1"], "p_L2": cal["L2"], "p_IF": cal.get("IF", np.full(n, np.nan)),
            "fused": fused, "risk": risk, "score_level": score_level, "l0_level": l0r.min_level,
            "l0_log_flag": l0r.log_flags, "abstained": abstained, "level": level, "action": actions,
            "band": [BANDS[a] for a in actions],
            "treatment": [TXN_TREATMENT.get(t, TXN_TREATMENT["pay"]).get(a, "none") for t, a in zip(ttype, actions)],
            "reasons": reasons, "l0_fired": l0_ids, "reason_fallback": fallback,
        })

    def message(self, action: str, reasons: list[str], txn_type: str, lang: str = "en",
                amount: float | None = None) -> dict:
        return customer_message(self.messages, action, reasons, txn_type, lang, amount)

    # --- io -------------------------------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written bundle
        return path


def _codes_from_text(text: str) -> dict:
    from .l0 import _parse
    codes = yaml.safe_load(text)["codes"]
    for code, spec in codes.items():
        if spec.get("predicate"):
            spec["_tree"], _ = _parse(code, spec["predicate"])
    return codes


def load_bundle(path: Path) -> Bundle:
    with open(path, "rb") as fh:
        b = pickle.load(fh)
    if getattr(b, "version", None) != BUNDLE_VERSION:
        raise ValueError(f"bundle version {getattr(b, 'version', None)} != {BUNDLE_VERSION}")
    return b


def model_card(b: Bundle) -> dict[str, Any]:
    return {
        "name": "Citadel UPI Scam-Sequence Shield",
        "run_id": b.provenance.get("run_id"), "provenance": b.provenance,
        "intended_use": "PSP/bank-side risk scoring of UPI payments at the moment of approval; graded friction, never decline.",
        "not_for": ["automatic declines", "credit or account decisions", "use on real data without revalidation"],
        "training_data": "100% synthetic (Citadel generator); see docs/DISCLOSURE.md",
        "channels": {"L0": f"{len(b.rules)} deterministic rule cards", "L1": f"HistGradientBoosting, {b.l1.n_iter} rounds",
                     "L2": "logistic payee/mule-graph model", "IF": "isolation forest" if b.iforest else "not used"},
        "fusion_arm": b.fusion.arm,
        "ladder_thresholds_on_risk": b.ladder.thresholds, "ladder_budgets": b.ladder.budgets,
        "ladder_realised_shares_stats_slice": b.ladder.realised_shares,
        "abstention": b.abstain, "n_features": len(b.features),
        "reason_codes": sorted(b.codes), "languages": b.messages.get("languages", []),
    }
