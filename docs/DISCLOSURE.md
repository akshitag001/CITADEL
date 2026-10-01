# Disclosure

The full, generated disclosure (with run ids, library versions and assumptions) is
`docs/round1/appendix_disclosure.md`, rebuilt by `python -m citadel export-round1`.

Summary:
- **Data:** 100% synthetic, produced by Citadel's own generator. No real bank, customer or platform data; nothing scraped.
- **Models:** scikit-learn (gradient-boosted trees, logistic regression, isolation forest, isotonic calibration). No pretrained models. No LLM in the scoring path.
- **Reused components:** none. All code was written for Citadel; runtime libraries are listed in `requirements.txt`.
- **AI-generated content:** code, configuration and documentation were written with an AI coding assistant (Claude Code) under the team's direction. All metrics are produced by the pipeline and checked by `scripts/verify_numbers.py`.
- **Assumptions:** the cost model, heed rates, label latency, analyst capacity and signal observability are labelled assumptions and are swept or stated wherever used.
