"""Re-derive every number in docs/round1/numbers.json from the run's CSVs; fail on any drift.

    python scripts/verify_numbers.py [--run demo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citadel.report import numbers as NUM  # noqa: E402
from citadel.runctx import make_ctx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="demo")
    a = ap.parse_args()
    ctx = make_ctx(None, a.run)
    art = ctx.art
    fresh = NUM.build_numbers(art, json.loads((art / "generate_summary.json").read_text()),
                              json.loads((art / "generate_fidelity_summary.json").read_text()), ctx.reportable())
    saved = ROOT / "docs" / "round1" / "numbers.json"
    problems = NUM.verify(saved, fresh)
    brief = (ROOT / "docs" / "round1" / "PDF_BRIEF.md").read_text(encoding="utf-8")
    nums = json.loads(saved.read_text(encoding="utf-8"))
    unrep = [k for k, v in nums.items() if not v["reportable"] and v["display"] in brief and len(v["display"]) > 3]
    problems += [f"{k}: not reportable but appears in PDF_BRIEF.md" for k in unrep]
    if problems:
        print("FAIL\n" + "\n".join(problems))
        return 1
    print(f"OK: {len(nums)} numbers match artifacts/{a.run}/ (run {ctx.prov.run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
