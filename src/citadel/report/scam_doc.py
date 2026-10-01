"""docs/round1/SCAM_SEQUENCE.md, generated from configs/scam_library.yaml (never hand-typed)."""

from __future__ import annotations

from ..config import ROOT
from ..generate.library import load_library


def _sig(spec: dict) -> str:
    extra = ", ".join(f"{k} {v}" for k, v in spec.items() if k != "p")
    return f"p={spec['p']}" + (f" ({extra})" if extra else "")


def write_scam_sequence_doc() -> str:
    lib = load_library()
    L = ["# Scam sequence grammar\n\n<!-- generated from configs/scam_library.yaml by `python -m citadel scam-doc` -->\n\n",
         "Every scam is a four-step sequence: **S1 lure** (unobserved) -> **S2 pressure / remote access** -> "
         "**S3 collect or pay a new payee (Citadel acts here)** -> **S4 mule fan-out**. Signals fire "
         "probabilistically; unobservable signals stay missing.\n\n",
         "| Variant | S3 type | Payments | Amount | First-time payee | S2 signals | S3 signals | Segments |\n",
         "|---|---|---|---|---|---|---|---|\n"]
    for k, v in lib.variants.items():
        s2, s3 = v["steps"]["S2"], v["steps"]["S3"]
        amt = s3["amount"]
        amt_s = {"baseline": f"{amt.get('multiple')}x usual", "balance": f"{amt.get('share')} of balance",
                 "absolute": f"INR {amt.get('inr')}"}[amt["base"]]
        L.append(f"| **{k}** {v['title']} | {s3['txn_type']} | {s3['payments']} | {amt_s} | {s3['first_time_payee_p']} | "
                 + "<br>".join(f"{c}: {_sig(sp)}" for c, sp in (s2.get("signals") or {}).items()) + " | "
                 + "<br>".join(f"{c}: {_sig(sp)}" for c, sp in (s3.get("signals") or {}).items()) + " | "
                 + ", ".join(v["allowed_segments"]) + " |\n")
    L.append("\n## Columns each step can influence\n\n| Variant | S2 | S3 | S4 |\n|---|---|---|---|\n")
    for k in lib.variants:
        sc = lib.step_columns(k)
        L.append(f"| {k} | {', '.join(sc['S2']) or '—'} | {', '.join(sc['S3'])} | {', '.join(sc['S4'])} |\n")
    L.append("\n## Evasions (attacker-controllable levers only)\n\n| Evasion | Levers | Requires |\n|---|---|---|\n")
    for e in lib.evasions.values():
        req = e.get("requires", {})
        L.append(f"| {e['title']} | {', '.join(e['levers'])} | {req.get('reason', '—') if req else '—'} |\n")
    space = lib.composition_space()
    L.append(f"\n## Composition space\n\n{space['size_legal']} legal of {space['size_total']} (variant x evasion x "
             f"segment). Every illegal combination is explained by the constraint it fails, e.g.:\n\n")
    seen = set()
    for r in space["illegal"]:
        key = r["reason"].split(";")[0][:60]
        if key in seen:
            continue
        seen.add(key)
        L.append(f"- {r['variant']} + {r['evasion']} + {r['segment']}: {r['reason']}\n")
    text = "".join(L)
    out = ROOT / "docs" / "round1" / "SCAM_SEQUENCE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
