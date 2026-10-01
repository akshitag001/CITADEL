# Threat model (STRIDE over the data flow)

Data flow: UPI app SDK -> scoring API -> feature state (switch history, profiles, graph) -> bundle (L0/L1/L2,
ladder) -> customer response / analyst store -> feedback.

| # | Element | Threat (STRIDE) | Mitigation | Test |
|---|---|---|---|---|
| T1 | scoring API | **S**poofing a client to see analyst data | bearer tokens per role; analyst endpoints 403 for customers | `test_customer_cannot_reach_analyst_endpoints`, `test_scoring_requires_a_token` |
| T2 | scoring API | **I**nfo disclosure: fraudster maps the decision boundary by probing | customer payload is band + reasons only (4 bands = 2 bits/query); per-token rate limit | `test_customer_payload_never_contains_a_score`, `test_boundary_probing_is_rate_limited`, `test_rate_limiter_window_slides` |
| T3 | request schema | **T**ampering: unexpected fields silently defaulted | pydantic `extra=forbid` -> 422; unknown accounts -> 422 | `test_unknown_feature_key_is_rejected_not_defaulted`, `test_unknown_account_is_rejected` |
| T4 | complaint feed | **T**ampering: false-report poisoning against innocent payees | complaints count only from distinct reporters aged >= 30 days; L0-01 also needs a structural mule signal; complaints alone cannot hold | `test_complaint_poisoning_cannot_reach_a_hold_alone`, `test_fresh_reporter_accounts_do_not_count`, `test_complaints_require_distinct_aged_reporters` |
| T5 | analyst store | **R**epudiation / **I**nfo disclosure: silent evidence browsing | every evidence view written to the audit log; evidence excludes ground truth | `test_evidence_access_is_audited`, `test_evidence_never_contains_ground_truth` |
| T6 | stored data | **I**nfo disclosure: identifiers in the store | salted HMAC ids (salt from env); no message content; retention TTL purge | `test_ids_are_hmac_hashed_with_a_salt`, `test_retention_ttl_purges_old_alerts` |
| T7 | model | **D**enial of service / model down -> everything approved | L0 runs with the model stack unavailable | `test_l0_runs_with_the_model_stack_unavailable` |
| T8 | model | adversarial adaptation (new script / evasion) | sealed holdout + leave-one-variant-out measured and published; Scam Studio adds scripts; retrain trigger below | `test_sealed_holdout_is_entity_level`, `eval_generalisation.csv`, `eval_lovo.csv` |
| T9 | training data | leakage that flatters the model | truth columns cannot reach features; prefix invariance; artefact hunter; canary | `test_no_truth_column_reaches_the_matrix`, `test_feature_matrix_is_prefix_invariant`, `test_no_fraud_only_id_namespace` |
| T10 | customer screen | **E**levation / harm: automatic block from one weak signal | no decline level exists; L0 rules need conjunctions and are zero-FP on legitimate populations; holds have release + appeal | `test_l0_zero_false_positives_on_legitimate_populations`, `test_held_payment_always_has_a_release_and_appeal_path`, `test_collapsed_ladder_is_refused` |
| T11 | consented signals | privacy: signal collected without consent | missing (NaN) without consent, never zero; NaN never satisfies a rule | `test_unobservable_signals_stay_missing_not_zero`, `test_nan_never_satisfies_a_rule` |

**Retraining trigger (T8).** Retrain when (a) recall on a newly confirmed script measured in the Studio falls
below the seen-variant recall by more than the generalisation gap on record, (b) a reason card's precision
drops below its last measured value, or (c) A3 volume exceeds analyst capacity for 3 consecutive days.

**Not covered:** compromised device/SDK lying about signals (mitigated only by never acting on one signal),
insider analyst collusion (audit log only), and real-world distribution shift (synthetic-only validation).
