# Citadel — one-command targets. On Windows use `mingw32-make <target>` (or run the python commands directly).
PY ?= python
PROFILE ?= default
RUN ?= demo
CFG = configs/$(PROFILE).yaml

.PHONY: install demo quick test test-fast lint serve round1 verify bundle clean

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# Full pipeline: generate -> fidelity -> features -> train -> evaluate -> REPORT.md
demo:
	$(PY) -m citadel run --config $(CFG) --run $(RUN)
	$(PY) scripts/build_replay_bundle.py --run $(RUN)

quick:
	$(PY) -m citadel run --config configs/quick.yaml --run quick

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

lint:
	$(PY) -m ruff check src tests scripts

serve:
	$(PY) -m citadel serve --run $(RUN)

round1:
	$(PY) -m citadel export-round1 --run $(RUN)

bundle:
	$(PY) scripts/build_replay_bundle.py --run $(RUN)

verify:
	$(PY) scripts/verify.py --run $(RUN)
