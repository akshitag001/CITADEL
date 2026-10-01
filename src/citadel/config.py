"""One typed configuration tree, loaded from YAML, plus key-derived RNG streams.

Determinism contract: the same resolved Config and the same seed produce a byte-identical dataset.
Every random component draws from its own stream, ``rng(cfg, "component/name")``, so adding a new
component never shifts the draws of an existing one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


@dataclass
class TimeCfg:
    tz: str = "Asia/Kolkata"
    start_date: str = "2026-06-01"


@dataclass
class PopulationCfg:
    users: int = 20000
    merchants: int = 4000
    new_user_share: float = 0.15
    new_merchant_share: float = 0.20
    informal_merchant_share: float = 0.02
    screen_signal_observable_share: float = 0.6
    family_device_share: float = 0.05


@dataclass
class TrafficCfg:
    days: int = 45
    txn_per_user_day: float = 0.35
    hard_negative_share: float = 0.04
    shape_share: float = 0.03
    festival_day: int = 30


@dataclass
class AttacksCfg:
    target_share: float = 0.005
    variants: list[str] = field(default_factory=lambda: ["V1", "V2", "V3", "V4", "V5", "V6"])
    variant_weights: dict[str, float] = field(default_factory=lambda: {
        "V1": 1.0, "V2": 0.8, "V3": 1.0, "V4": 0.8, "V5": 0.9, "V6": 0.7})
    evasion_share: float = 0.25
    night_bias: float = 1.5
    mules_per_episode: float = 0.3
    label_visible_share: float = 0.45


@dataclass
class SealedCfg:
    variant: str = "V4"
    evasion: str = "signal_suppression"


@dataclass
class SplitCfg:
    train_share: float = 0.52
    purge_days: float = 3.0
    calibration_share: float = 0.12
    stats_share: float = 0.09
    embargo_days: float = 2.0
    inner_valid_share: float = 0.15


@dataclass
class LadderCfg:
    A1_max_share: float = 0.03
    A2_max_share: float = 0.010
    A3_max_share: float = 0.001
    abstain_max_share: float = 0.0005
    abstain_disagreement: float = 0.5


@dataclass
class StaffingCfg:
    analysts: int = 4
    cases_per_hour: float = 12.0
    shift_hours: float = 8.0
    shifts_per_day: int = 2
    deployment_payments_per_day: int = 500000


@dataclass
class ModelCfg:
    max_iter: int = 400
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    positive_weight: float = 10.0


@dataclass
class EvalCfg:
    fpr_budgets: list[float] = field(default_factory=lambda: [0.001, 0.005])
    min_positives_for_headline: int = 200
    min_positives_for_cell: int = 20
    bootstrap_resamples: int = 300
    lovo: bool = True
    sample_size_fracs: list[float] = field(default_factory=lambda: [0.125, 0.25, 0.5, 1.0])


@dataclass
class ConsentCfg:
    call_in_progress_available_share: float = 0.35


@dataclass
class ServeCfg:
    customer_rate_per_min: int = 60
    analyst_rate_per_min: int = 600
    retention_days: int = 180
    hmac_salt_env: str = "CITADEL_HMAC_SALT"


@dataclass
class Config:
    seed: int = 7
    run_name: str = "demo"
    profile: str = "default"
    time: TimeCfg = field(default_factory=TimeCfg)
    population: PopulationCfg = field(default_factory=PopulationCfg)
    traffic: TrafficCfg = field(default_factory=TrafficCfg)
    attacks: AttacksCfg = field(default_factory=AttacksCfg)
    sealed_holdout: SealedCfg = field(default_factory=SealedCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    ladder_budgets: LadderCfg = field(default_factory=LadderCfg)
    staffing: StaffingCfg = field(default_factory=StaffingCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    consent: ConsentCfg = field(default_factory=ConsentCfg)
    serve: ServeCfg = field(default_factory=ServeCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    """Construct a dataclass from a dict, rejecting unknown keys (a typo must fail, not be ignored)."""
    if data is None:
        return cls()
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"unknown config key(s) at {path or 'root'}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        f = names[key]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            kwargs[key] = _build(type(default), value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("inherits", None)
    if parent:
        data = _deep_merge(_read_yaml(path.parent / parent), data)
    return data


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    data = _read_yaml(path)
    data = _deep_merge(data, overrides)
    return _build(Config, data, "")


def stable_hash(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def rng(cfg_or_seed: Config | int, component: str) -> np.random.Generator:
    """A generator derived from (seed, stable hash of the component name)."""
    seed = cfg_or_seed.seed if isinstance(cfg_or_seed, Config) else int(cfg_or_seed)
    return np.random.default_rng(np.random.SeedSequence([seed, stable_hash(component)]))


def config_path(name: str) -> Path:
    return CONFIG_DIR / name
