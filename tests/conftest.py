"""Shared fixtures. The tiny profile is generated once per session; the trained tiny run is cached
under artifacts/pytest_tiny (gitignored) and rebuilt whenever its config, library or code changes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from citadel.config import load_config  # noqa: E402

TINY = ROOT / "configs" / "tiny.yaml"


@pytest.fixture(scope="session")
def tiny_cfg():
    return load_config(TINY)


@pytest.fixture(scope="session")
def tiny_gen(tiny_cfg):
    from citadel.generate.campaign import generate
    return generate(tiny_cfg)


@pytest.fixture(scope="session")
def tiny_features(tiny_gen):
    from citadel.features.matrix import build_matrix, known_mules_from_labels, profiles_from_accounts
    return build_matrix(tiny_gen.transactions, profiles_from_accounts(tiny_gen.population.accounts),
                        known_mules_from_labels(tiny_gen.labels))


@pytest.fixture(scope="session")
def tiny_run():
    """A fully trained + evaluated tiny run (generate -> features -> fidelity -> train -> evaluate)."""
    from citadel import pipeline as P
    from citadel.provenance import hash_files
    from citadel.runctx import make_ctx
    ctx = make_ctx(TINY, "pytest_tiny")
    stamp = ctx.art / ".stamp"
    key = ctx.prov.run_id + hash_files(list((ROOT / "src").rglob("*.py")))
    if not (stamp.exists() and stamp.read_text() == key and (ctx.art / "REPORT.md").exists()):
        P.stage_all(ctx)
        stamp.write_text(key)
    return ctx


@pytest.fixture(scope="session")
def tiny_state(tiny_run):
    from citadel.serve.state import load_state
    return load_state(tiny_run.run)
