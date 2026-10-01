"""Loader and validator for the scam grammar (configs/scam_library.yaml).

The grammar is data. This module refuses to load a library that names a column the schema does not
have, and explains — with the failed constraint, not a boolean — why a composition is illegal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import CONFIG_DIR
from ..schema import COLUMNS, TRUTH_ONLY

SIGNAL_FORMS = {"p", "lognormal", "uniform", "int_range", "value"}
AMOUNT_BASES = {"baseline", "balance", "absolute"}
SEGMENT_FILTERS = {"title", "age_band", "digital_literacy", "max_tenure_days"}
VICTIM_WEIGHT_KEYS = {"user_age_band", "digital_literacy", "home_tier", "is_seller"}


class LibraryError(ValueError):
    pass


@dataclass
class ScamLibrary:
    raw: dict[str, Any]

    @property
    def variants(self) -> dict[str, dict[str, Any]]:
        return self.raw["variants"]

    @property
    def evasions(self) -> dict[str, dict[str, Any]]:
        return self.raw["evasions"]

    @property
    def segments(self) -> dict[str, dict[str, Any]]:
        return self.raw["segments"]

    def variant_signal_columns(self, vid: str) -> set[str]:
        steps = self.variants[vid]["steps"]
        cols: set[str] = set()
        for s in ("S2", "S3"):
            cols |= set((steps.get(s) or {}).get("signals", {}) or {})
        return cols

    def variant_txn_types(self, vid: str) -> set[str]:
        return {self.variants[vid]["steps"]["S3"]["txn_type"]}

    def step_columns(self, vid: str) -> dict[str, list[str]]:
        """Which schema columns each step can influence (for docs and the Studio)."""
        steps = self.variants[vid]["steps"]
        out = {"S1": []}
        s2 = sorted((steps["S2"].get("signals") or {}).keys())
        if steps["S2"].get("prelim", {}).get("p", 0) > 0:
            s2 = sorted(set(s2) | {"amount_inr", "payee_id"})
        out["S2"] = s2
        out["S3"] = sorted(set((steps["S3"].get("signals") or {}).keys())
                           | {"amount_inr", "payee_id", "txn_type", "payee_name_similarity_to_known_contact"})
        out["S4"] = ["payee_inbound_unique_payers_24h", "payee_inbound_amount_24h", "(mule outbound rows)"]
        return out

    # --- composition space ---------------------------------------------------------------------
    def explain_illegal(self, variant: str, evasion: str | None, segment: str) -> str | None:
        """None when legal; otherwise the specific constraint that failed."""
        if variant not in self.variants:
            return f"unknown variant '{variant}'"
        if segment not in self.segments:
            return f"unknown segment '{segment}'"
        v = self.variants[variant]
        if segment not in v["allowed_segments"]:
            return (f"segment '{segment}' is not targeted by {variant} ({v['title']}); allowed: "
                    f"{', '.join(v['allowed_segments'])}")
        if evasion in (None, "", "none"):
            return None
        if evasion not in self.evasions:
            return f"unknown evasion '{evasion}'"
        req = self.evasions[evasion].get("requires") or {}
        if "signals_any" in req and not (set(req["signals_any"]) & self.variant_signal_columns(variant)):
            return f"evasion '{evasion}' is illegal for {variant}: {req.get('reason', 'requirement failed')}"
        if "txn_types_any" in req and not (set(req["txn_types_any"]) & self.variant_txn_types(variant)):
            return f"evasion '{evasion}' is illegal for {variant}: {req.get('reason', 'requirement failed')}"
        return None

    def compatible_evasions(self, variant: str) -> list[str]:
        return [e for e in self.evasions if self.explain_illegal(variant, e, "mainstream") is None]

    def composition_space(self) -> dict[str, Any]:
        legal, illegal = [], []
        for v, e, s in itertools.product(self.variants, ["none", *self.evasions], self.segments):
            why = self.explain_illegal(v, e, s)
            (illegal if why else legal).append({"variant": v, "evasion": e, "segment": s, "reason": why})
        return {"size_total": len(legal) + len(illegal), "size_legal": len(legal),
                "legal": legal, "illegal": illegal}


def _check_signal(where: str, col: str, spec: Any) -> None:
    if col not in COLUMNS:
        raise LibraryError(f"{where}: signal '{col}' is not a schema column")
    if COLUMNS[col].role == TRUTH_ONLY:
        raise LibraryError(f"{where}: signal '{col}' is TRUTH_ONLY and cannot be influenced by a scam")
    if not isinstance(spec, dict) or "p" not in spec:
        raise LibraryError(f"{where}: signal '{col}' needs a dict with a probability 'p'")
    if not set(spec) <= SIGNAL_FORMS:
        raise LibraryError(f"{where}: signal '{col}' has unknown keys {sorted(set(spec) - SIGNAL_FORMS)}")
    if not 0.0 <= float(spec["p"]) <= 1.0:
        raise LibraryError(f"{where}: signal '{col}' probability out of [0,1]")
    if float(spec["p"]) == 1.0 and set(spec) == {"p"}:
        raise LibraryError(f"{where}: flag '{col}' fires always (p=1); signals must be probabilistic")


def _check_amount(where: str, spec: dict[str, Any]) -> None:
    if spec.get("base") not in AMOUNT_BASES:
        raise LibraryError(f"{where}: amount base must be one of {sorted(AMOUNT_BASES)}")


def validate(raw: dict[str, Any]) -> None:
    for key in ("variants", "evasions", "segments", "attacker_levers", "victim_immutable"):
        if key not in raw:
            raise LibraryError(f"library missing top-level key '{key}'")
    for col in raw["victim_immutable"]:
        if col not in COLUMNS:
            raise LibraryError(f"victim_immutable: '{col}' is not a schema column")
    for name, seg in raw["segments"].items():
        if not set(seg) <= SEGMENT_FILTERS:
            raise LibraryError(f"segment {name}: unknown filters {sorted(set(seg) - SEGMENT_FILTERS)}")
    levers = set(raw["attacker_levers"])
    for vid, v in raw["variants"].items():
        steps = v.get("steps") or {}
        if set(steps) != {"S1", "S2", "S3", "S4"}:
            raise LibraryError(f"{vid}: steps must be exactly S1..S4")
        for s in ("S2", "S3"):
            for col, spec in (steps[s].get("signals") or {}).items():
                _check_signal(f"{vid}.{s}", col, spec)
        _check_amount(f"{vid}.S3", steps["S3"]["amount"])
        prelim = steps["S2"].get("prelim") or {}
        if prelim.get("p", 0) > 0:
            _check_amount(f"{vid}.S2.prelim", prelim["amount"])
        if steps["S3"]["txn_type"] not in COLUMNS["txn_type"].categories:
            raise LibraryError(f"{vid}: S3 txn_type invalid")
        for seg in v["allowed_segments"]:
            if seg not in raw["segments"]:
                raise LibraryError(f"{vid}: unknown segment '{seg}'")
        if not set(v.get("victim_weights", {})) <= VICTIM_WEIGHT_KEYS:
            raise LibraryError(f"{vid}: victim_weights keys must be in {sorted(VICTIM_WEIGHT_KEYS)}")
    for eid, e in raw["evasions"].items():
        bad = set(e.get("levers", [])) - levers
        if bad:
            raise LibraryError(f"evasion {eid}: levers {sorted(bad)} are not attacker-controllable")
        for col in e.get("suppress", []):
            if col not in COLUMNS:
                raise LibraryError(f"evasion {eid}: suppress column '{col}' is not a schema column")


def load_library(path: str | Path | None = None, raw: dict[str, Any] | None = None) -> ScamLibrary:
    if raw is None:
        path = Path(path) if path else CONFIG_DIR / "scam_library.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate(raw)
    return ScamLibrary(raw)
