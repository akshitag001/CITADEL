"""`make verify`: everything a reviewer would check, as one PASS/FAIL table.

    python scripts/verify.py [--run demo]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    return name, bool(ok), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="demo")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    import pandas as pd

    from citadel.runctx import make_ctx
    ctx = make_ctx(None, a.run)
    art = ctx.art
    rows = []
    if not a.skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                           capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        rows.append(check("test suite (all gates)", r.returncode == 0, last))
    fid = json.loads((art / "generate_fidelity_summary.json").read_text())
    rows.append(check("single-feature AUC < 0.95", fid["single_feature_passes"], f"max {fid['single_feature_max_auc']:.3f}"))
    rows.append(check("artefact hunter: no fraud-only namespace", fid["artefacts_found"] == 0, str(fid["artefacts_found"])))
    rows.append(check("derived probe below 'measuring the generator' ceiling",
                      not fid["derived_joint_probe"]["flag_measuring_generator"],
                      f"{fid['derived_joint_probe']['recall_at_fpr']:.3f} vs 0.92"))
    rows.append(check("label-shuffle null at chance", fid["label_shuffle_null"]["passes"],
                      f"ROC {fid['label_shuffle_null']['roc_auc']:.3f}"))
    rows.append(check("leakage canary detected", fid["leakage_canary"]["passes"], ""))
    h = pd.read_csv(art / "defend_headline.csv")
    rows.append(check("headline reportable (>= min positives, default profile)", bool(h.reportable.all()),
                      f"n_pos {int(h.n_positives.iloc[0])}"))
    rows.append(check("no accuracy figure", not h.metric.str.contains("accuracy").any()))
    lad = pd.read_csv(art / "defend_ladder_thresholds.csv")
    t = lad.set_index("action")["threshold_on_risk"]
    rows.append(check("ladder strictly monotonic", t["A1"] < t["A2"] < t["A3"]))
    b = pd.read_csv(art / "defend_alert_budget.csv").iloc[0]
    rows.append(check("warnings within budget", b.warnings_A2plus_per_1000 <= b.warning_budget_per_1000 * 1.1,
                      f"{b.warnings_A2plus_per_1000:.2f} / {b.warning_budget_per_1000:.0f} per 1,000"))
    rows.append(check("holds within analyst capacity", bool(b.holds_within_capacity),
                      f"{b.holds_A3_per_1000:.2f} vs {b.analyst_capacity_k_per_1000:.2f} per 1,000 (D-018)"))
    r = subprocess.run([sys.executable, "scripts/verify_numbers.py", "--run", a.run], cwd=ROOT, capture_output=True, text=True)
    rows.append(check("numbers index matches CSVs", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else ""))
    rows.append(check("replay bundle present", (ROOT / "frontend" / "replay" / "replay.js").exists()))
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8") + (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    import re
    urls = [u for u in re.findall(r"https?://[^\"' )>]+", js)
            if not u.startswith(("http://www.w3.org/", "http://127.0.0.1"))]  # xmlns ids are not requests
    rows.append(check("console makes no external requests", not urls, ", ".join(urls[:3])))
    print(f"\nCitadel verify — run {ctx.prov.run_id} ({a.run})\n" + "-" * 78)
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL':5} {name:58} {detail}")
    fails = [r for r in rows if not r[1]]
    print("-" * 78 + f"\n{len(rows) - len(fails)}/{len(rows)} passed"
          + ("  (known open risk: holds above capacity, D-018)" if any("capacity" in f[0] for f in fails) else ""))
    hard = [f for f in fails if "capacity" not in f[0]]
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
