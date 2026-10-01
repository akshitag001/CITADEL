"""The canonical UPI event schema: the contract between the generator and the defence.

Every column has exactly one role:

* ``ID``          identifiers and metadata. Never model inputs.
* ``FEATURE_OK``  what an institution could observe at decision time. May feed features.
* ``TRUTH_ONLY``  ground truth for evaluation. Must never reach a feature matrix (tests enforce this).

Nullable columns mean "the institution cannot observe this" (no SDK signal, no consent). They are
missing, NOT zero, and the model handles NaN natively.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ID, FEATURE_OK, TRUTH_ONLY = "ID", "FEATURE_OK", "TRUTH_ONLY"


@dataclass(frozen=True)
class Col:
    role: str
    dtype: str
    meaning: str
    nullable: bool = False
    categories: tuple[str, ...] = ()


COLUMNS: dict[str, Col] = {
    # --- identifiers -------------------------------------------------------------------------
    "txn_id": Col(ID, "str", "opaque transaction id"),
    "ts": Col(ID, "int64", "event time, Unix seconds UTC (hour/dow derived via timeutil, IST)"),
    "user_id": Col(ID, "str", "payer account, salted hash; same namespace as payees"),
    "payee_id": Col(ID, "str", "payee account, salted hash; same namespace as payers"),
    "device_id": Col(ID, "str", "payer device, hashed"),
    "session_id": Col(ID, "str", "app session, hashed"),
    "consent_call_signal": Col(ID, "int8", "user consented to the call-in-progress signal (metadata)"),
    # --- transaction -------------------------------------------------------------------------
    "txn_type": Col(FEATURE_OK, "category", "pay | collect | qr_pay", categories=("pay", "collect", "qr_pay")),
    "amount_inr": Col(FEATURE_OK, "float64", "payment amount in INR"),
    "hour": Col(FEATURE_OK, "int8", "IST hour of day"),
    "dow": Col(FEATURE_OK, "int8", "IST day of week, Monday=0"),
    # --- payer -------------------------------------------------------------------------------
    "user_age_band": Col(FEATURE_OK, "category", "payer age band",
                         categories=("18-25", "26-40", "41-60", "60+")),
    "user_tenure_days": Col(FEATURE_OK, "float64", "days since payer account opened"),
    "digital_literacy": Col(FEATURE_OK, "category", "payer digital literacy band",
                            categories=("low", "med", "high")),
    "balance_band": Col(FEATURE_OK, "category", "payer balance band", categories=("low", "mid", "high")),
    "account_balance_inr": Col(FEATURE_OK, "float64", "approximate available balance (issuer-bank only)"),
    "home_tier": Col(FEATURE_OK, "int8", "payer home city tier 1/2/3"),
    # --- payee -------------------------------------------------------------------------------
    "payee_age_days": Col(FEATURE_OK, "float64", "days since payee account opened"),
    "user_to_payee_prior_txn_count": Col(FEATURE_OK, "int32", "prior payments payer->payee"),
    "payee_inbound_unique_payers_24h": Col(FEATURE_OK, "int32", "distinct payers into payee, trailing 24h"),
    "payee_inbound_amount_24h": Col(FEATURE_OK, "float64", "INR into payee, trailing 24h"),
    "payee_complaint_count": Col(FEATURE_OK, "int32",
                                 "corroborated complaints on payee arrived before ts (distinct, aged reporters)"),
    "payee_is_verified_merchant": Col(FEATURE_OK, "int8", "payee is a verified merchant"),
    "payee_name_similarity_to_known_contact": Col(FEATURE_OK, "float64",
                                                  "max name similarity of payee to payer's saved contacts"),
    # --- device / session --------------------------------------------------------------------
    "device_age_days": Col(FEATURE_OK, "float64", "days since device first seen"),
    "new_device_flag": Col(FEATURE_OK, "int8", "payer bound this device < 24h ago"),
    "sim_change_7d": Col(FEATURE_OK, "int8", "SIM change on payer number in trailing 7d"),
    "session_duration_s": Col(FEATURE_OK, "float64", "app session length before approval"),
    "attempts_in_session": Col(FEATURE_OK, "int8", "payment attempts in this session"),
    "collect_request_age_s": Col(FEATURE_OK, "float64", "collect only: seconds request->approval",
                                 nullable=True),
    "request_source_known_contact": Col(FEATURE_OK, "float64", "collect only: requester is a known contact",
                                        nullable=True),
    "screen_share_active": Col(FEATURE_OK, "float64", "screen sharing active (SDK-observable only)",
                               nullable=True),
    "remote_access_app_detected": Col(FEATURE_OK, "float64", "remote-control app running (SDK-observable)",
                                      nullable=True),
    "call_in_progress": Col(FEATURE_OK, "float64", "phone call in progress (consented users only)",
                            nullable=True),
    # --- truth (evaluation only) -------------------------------------------------------------
    "episode_id": Col(TRUTH_ONLY, "str", "scam episode id, empty for benign", nullable=True),
    "is_attack": Col(TRUTH_ONLY, "int8", "victim payment to a scammer (step 2/3)"),
    "scam_variant": Col(TRUTH_ONLY, "str", "V1..V6 or empty", nullable=True),
    "evasion_technique": Col(TRUTH_ONLY, "str", "evasion name or empty", nullable=True),
    "step_in_sequence": Col(TRUTH_ONLY, "int8", "0 benign, 2 preliminary, 3 harmful payment, 4 mule fan-out"),
    "label_visible": Col(TRUTH_ONLY, "int8", "a label ever reaches the institution"),
    "label_delay_days": Col(TRUTH_ONLY, "float64", "days until the label arrives", nullable=True),
    "row_kind": Col(TRUTH_ONLY, "str", "benign|hard_negative|shape|seasoning|attack|mule_outflow|lure_credit"),
    "source_txn_id": Col(TRUTH_ONLY, "str", "benign row this attack row was mutated from", nullable=True),
    "touched_fields": Col(TRUTH_ONLY, "str", "comma list of fields the scam overwrote", nullable=True),
    "hn_archetype": Col(TRUTH_ONLY, "str", "hard-negative / shape archetype(s)", nullable=True),
}

#: Rows initiated by the criminal's own account. They are scored by the payee-side ops path, not the
#: customer ladder, and are excluded from customer-ladder metrics (DECISIONS.md D-004).
MULE_INITIATED_KINDS = ("mule_outflow", "lure_credit")


def columns_with_role(role: str) -> list[str]:
    return [c for c, spec in COLUMNS.items() if spec.role == role]


def feature_ok_columns() -> list[str]:
    return columns_with_role(FEATURE_OK)


def truth_columns() -> list[str]:
    return columns_with_role(TRUTH_ONLY)


def id_columns() -> list[str]:
    return columns_with_role(ID)


def categorical_columns() -> list[str]:
    return [c for c, s in COLUMNS.items() if s.dtype == "category"]


def nullable_columns() -> list[str]:
    return [c for c, s in COLUMNS.items() if s.nullable]


def validate_frame(df: pd.DataFrame) -> None:
    """Raise if the frame does not carry exactly the schema columns."""
    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={missing} extra={extra}")
    for c, spec in COLUMNS.items():
        if not spec.nullable and spec.role != TRUTH_ONLY and df[c].isna().any():
            raise ValueError(f"non-nullable column {c} has nulls")
    for c in categorical_columns():
        bad = set(df[c].astype(str).unique()) - set(COLUMNS[c].categories)
        if bad:
            raise ValueError(f"column {c} has values outside its categories: {sorted(bad)}")
