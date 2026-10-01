"""Run context: where a run's data and artefacts live, and its provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ROOT, Config, load_config
from .provenance import Provenance, make_provenance, write_csv, write_json


@dataclass
class RunCtx:
    cfg: Config
    run: str
    prov: Provenance

    @property
    def data(self) -> Path:
        return ROOT / "data" / self.run

    @property
    def art(self) -> Path:
        p = ROOT / "artifacts" / self.run
        p.mkdir(parents=True, exist_ok=True)
        return p

    def csv(self, df: pd.DataFrame, name: str) -> Path:
        return write_csv(df, self.art / name, self.prov)

    def json(self, obj, name: str) -> Path:
        return write_json(obj, self.art / name, self.prov)

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.data / name)

    def gen_meta(self) -> dict:
        return json.loads((self.data / "generation_summary.json").read_text())

    def reportable(self) -> bool:
        return self.cfg.profile == "default"


def make_ctx(config: str | Path | None = None, run: str | None = None) -> RunCtx:
    cfg = load_config(config) if config else load_config()
    if run is None:
        run = cfg.run_name
    else:
        # a run directory remembers which config produced it
        saved = ROOT / "artifacts" / run / "config_resolved.json"
        if config is None and saved.exists():
            cfg = load_config(ROOT / "configs" / f"{json.loads(saved.read_text())['profile']}.yaml")
    cfg.run_name = run
    return RunCtx(cfg, run, make_provenance(cfg))
