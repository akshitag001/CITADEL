# Finale checklist (24–25 Oct, IIT Delhi)

## Before travelling
- [ ] Fresh clone on a second laptop: `pip install -r requirements.txt && pip install -e . && python -m citadel run --run demo` succeeds
- [ ] `python -m pytest -q` green; `python scripts/verify.py --run demo` shows only the disclosed capacity FAIL
- [ ] `python scripts/build_replay_bundle.py --run demo` re-recorded after the final run; committed
- [ ] Offline test: Wi-Fi off, open `frontend/index.html` from disk, every tab renders
- [ ] Demo timed at 3, 5 and 8 minutes (`docs/DEMO_SCRIPT.md`)
- [ ] Backup screen recording of the 5-minute demo on two USB sticks
- [ ] Printed one-page architecture (`docs/round1/figures/F1_architecture.png`) and numbers index
- [ ] Chargers, HDMI/USB-C dongles, extension cord, phone hotspot

## 48-hour build plan
1. Capacity guard: overflow holds become A2 warnings; re-measure (D-018)
2. Per-segment thresholds with an FPR-ratio cap for seniors; re-measure (D-019)
3. A new scam script added to `scam_library.yaml`, measured as unseen through the Studio
4. Stretch, in order: hashed mule exchange; consented on-device lure flag; lite red-team loop
5. Every new number goes through the pipeline -> numbers index; nothing typed

## Judge Q&A
- [ ] Everyone has read `docs/QA_BANK.md` and can open the evidence file for any answer
