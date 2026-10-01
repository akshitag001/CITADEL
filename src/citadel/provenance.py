"""What produced this number — stamped on every artefact and every API response.

A content-derived run id (seed + git sha + config hash + library hash), so two identical runs collide by design and a
reviewer can tell at a glance that a rerun reproduced.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG_DIR, ROOT, Config


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_sha() -> str:
    return _git("rev-parse", "--short=12", "HEAD") or "nogit"


def git_dirty() -> bool:
    if not _git("rev-parse", "--git-dir"):
        return False
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def hash_object(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def library_hash() -> str:
    """Hash of the scam grammar + rule/reason/message cards: inputs to every downstream number."""
    names = ["scam_library.yaml", "l0_rules.yaml", "reason_codes.yaml", "messages.yaml", "costs.yaml"]
    return hash_files([CONFIG_DIR / n for n in names])


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:  # pragma: no cover
            out[name] = "absent"
    return out


@dataclass
class Provenance:
    run_name: str
    seed: int
    git_sha: str
    git_dirty: bool
    config_hash: str
    library_hash: str
    environment: dict[str, str] = field(default_factory=dict)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self) -> str:
        dirty = " (dirty tree)" if self.git_dirty else ""
        return (f"run `{self.run_id}` | seed {self.seed} | code `{self.git_sha}`{dirty} | "
                f"config `{self.config_hash}` | library `{self.library_hash}`")


def make_provenance(cfg: Config) -> Provenance:
    p = Provenance(run_name=cfg.run_name, seed=cfg.seed, git_sha=git_sha(), git_dirty=git_dirty(),
                   config_hash=cfg.config_hash(), library_hash=library_hash(), environment=_versions())
    p.run_id = hash_object({"seed": p.seed, "git": p.git_sha, "cfg": p.config_hash, "lib": p.library_hash})[:12]
    return p


def stamp_dict(d: dict[str, Any], prov: Provenance) -> dict[str, Any]:
    out = dict(d)
    out["_provenance"] = {"run_id": prov.run_id, "config_hash": prov.config_hash,
                          "library_hash": prov.library_hash, "git_sha": prov.git_sha}
    return out


def write_csv(df: pd.DataFrame, path: Path, prov: Provenance) -> Path:
    """Every CSV carries the run id as a column so a number can never be separated from its run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["run_id"] = prov.run_id
    out.to_csv(path, index=False, lineterminator="\n")
    return path


def write_json(obj: Any, path: Path, prov: Provenance) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp_dict(obj, prov) if isinstance(obj, dict) else {"data": obj, **stamp_dict({}, prov)}
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o: Any) -> Any:
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
