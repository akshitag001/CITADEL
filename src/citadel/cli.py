"""python -m citadel <stage> [--config configs/x.yaml] [--run name]"""

from __future__ import annotations

import argparse
import sys

STAGES = ["generate", "features", "fidelity", "train", "evaluate", "run", "serve", "export-round1",
          "scam-doc"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="citadel", description="Citadel: UPI Scam-Sequence Shield")
    p.add_argument("stage", choices=STAGES, help="pipeline stage")
    p.add_argument("--config", default=None, help="config YAML (default: configs/default.yaml)")
    p.add_argument("--run", default=None, help="run name (default: config run_name)")
    p.add_argument("--quick", action="store_true", help="shortcut for --config configs/quick.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    if a.quick:
        a.config = "configs/quick.yaml"

    from . import pipeline as P
    from .runctx import make_ctx
    ctx = make_ctx(a.config, a.run)
    if a.stage == "generate":
        P.stage_generate(ctx)
    elif a.stage == "features":
        P.stage_features(ctx)
    elif a.stage == "fidelity":
        P.stage_fidelity(ctx)
    elif a.stage == "train":
        P.stage_train(ctx)
    elif a.stage == "evaluate":
        P.stage_evaluate(ctx)
    elif a.stage == "run":
        P.stage_all(ctx)
    elif a.stage == "serve":
        import uvicorn

        from .serve.api import create_app
        uvicorn.run(create_app(ctx.run), host=a.host, port=a.port)
    elif a.stage == "export-round1":
        from .report.round1 import export_round1
        export_round1(ctx)
    elif a.stage == "scam-doc":
        from .report.scam_doc import write_scam_sequence_doc
        write_scam_sequence_doc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
