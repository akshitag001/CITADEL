"""Every model feature, declared once: family, window, required columns, causality, availability.

``availability`` says whether a deployment position can construct the feature:
psp = payer's UPI app/PSP, issuer = payer's bank, wallet = closed-loop wallet. "partial" means only for
counterparties the institution itself serves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

FAMILIES = ("raw", "user", "sequence", "payee", "graph")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    window: str
    requires: tuple[str, ...]
    description: str
    psp: str = "full"
    issuer: str = "full"
    wallet: str = "full"
    causal: bool = True


def _f(name, family, window, requires, description, psp="full", issuer="full", wallet="full"):
    return FeatureSpec(name, family, window, tuple(requires), description, psp, issuer, wallet)


REGISTRY: list[FeatureSpec] = [
    # ---- raw schema (transformed only) -------------------------------------------------------------
    _f("f_amount_log", "raw", "event", ["amount_inr"], "log1p(amount)"),
    _f("f_txn_type", "raw", "event", ["txn_type"], "0 pay, 1 collect, 2 qr_pay"),
    _f("f_hour", "raw", "event", ["hour"], "IST hour"),
    _f("f_dow", "raw", "event", ["dow"], "IST day of week"),
    _f("f_age_band", "raw", "profile", ["user_age_band"], "payer age band ordinal"),
    _f("f_literacy", "raw", "profile", ["digital_literacy"], "payer literacy ordinal"),
    _f("f_tier", "raw", "profile", ["home_tier"], "home city tier"),
    _f("f_tenure_days", "raw", "profile", ["user_tenure_days"], "payer tenure"),
    _f("f_balance_band", "raw", "profile", ["balance_band"], "balance band ordinal",
       psp="none", wallet="partial"),
    _f("f_payee_age_days", "raw", "profile", ["payee_age_days"], "payee account age", psp="partial"),
    _f("f_payee_prior", "raw", "all history", ["user_to_payee_prior_txn_count"], "prior payments payer->payee"),
    _f("f_payee_first_time", "raw", "all history", ["user_to_payee_prior_txn_count"], "never paid this payee"),
    _f("f_payee_unique_payers_24h", "raw", "24h", ["payee_inbound_unique_payers_24h"],
       "distinct payers into payee", psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_inbound_amt_24h", "raw", "24h", ["payee_inbound_amount_24h"], "INR into payee",
       psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_complaints", "raw", "all history", ["payee_complaint_count"], "corroborated complaints"),
    _f("f_payee_verified", "raw", "profile", ["payee_is_verified_merchant"], "verified merchant"),
    _f("f_payee_name_sim", "raw", "event", ["payee_name_similarity_to_known_contact"], "name mimicry score"),
    _f("f_device_age_days", "raw", "profile", ["device_age_days"], "device age"),
    _f("f_new_device", "raw", "24h", ["new_device_flag"], "device bound < 24h ago"),
    _f("f_sim_change", "raw", "7d", ["sim_change_7d"], "SIM change in 7d", psp="partial"),
    _f("f_session_duration", "raw", "event", ["session_duration_s"], "session length"),
    _f("f_attempts", "raw", "event", ["attempts_in_session"], "attempts in session"),
    _f("f_collect_age_s", "raw", "event", ["collect_request_age_s"], "collect request age (NaN if not collect)"),
    _f("f_req_known_contact", "raw", "event", ["request_source_known_contact"], "collect from known contact"),
    _f("f_screen_share", "raw", "event", ["screen_share_active"], "screen share (NaN = unobservable)",
       issuer="none"),
    _f("f_remote_access", "raw", "event", ["remote_access_app_detected"], "remote-access app (NaN = unobservable)",
       issuer="none"),
    _f("f_call", "raw", "event", ["call_in_progress"], "call in progress (NaN = no consent)", issuer="none"),
    # ---- user baseline deviation ---------------------------------------------------------------------
    _f("f_user_txn_count_prior", "user", "all history", ["user_id", "ts"], "payer's prior payments (incl. profile)"),
    _f("f_is_cold_start", "user", "all history", ["user_id", "ts"], "fewer than 5 prior payments"),
    _f("f_amt_z_user", "user", "all history", ["amount_inr", "user_id"], "z-score of log amount vs payer"),
    _f("f_amt_ratio_user", "user", "all history", ["amount_inr", "user_id"], "amount / payer's typical"),
    _f("f_amt_over_user_p99", "user", "all history", ["amount_inr", "user_id"], "above payer's p99 estimate"),
    _f("f_share_of_balance", "user", "event", ["amount_inr", "account_balance_inr"], "amount / balance",
       psp="none", wallet="partial"),
    _f("f_user_count_1h", "user", "1h", ["user_id", "ts"], "payer payments in 1h"),
    _f("f_user_count_24h", "user", "24h", ["user_id", "ts"], "payer payments in 24h"),
    _f("f_user_amount_24h", "user", "24h", ["user_id", "amount_inr"], "payer INR in 24h"),
    _f("f_user_newpayee_rate", "user", "all history", ["user_id", "user_to_payee_prior_txn_count"],
       "payer's historical new-payee rate"),
    _f("f_user_txn_type_share", "user", "all history", ["user_id", "txn_type"],
       "share of payer's history with this txn_type"),
    _f("f_hour_dev_user", "user", "all history", ["user_id", "hour"], "circular distance from usual hour"),
    _f("f_session_ratio_user", "user", "all history", ["user_id", "session_duration_s"],
       "session length / payer's typical"),
    # ---- sequence (the heart of the product) ------------------------------------------------------------
    _f("f_secs_since_anomaly", "sequence", "all history", ["user_id", "new_device_flag", "sim_change_7d",
                                                            "screen_share_active", "remote_access_app_detected"],
       "seconds since payer's last session anomaly"),
    _f("f_anomaly_then_newpayee", "sequence", "60m", ["user_id", "user_to_payee_prior_txn_count"],
       "anomaly now or in the last hour AND first-time payee"),
    _f("f_newpayee_count_1h", "sequence", "1h", ["user_id", "user_to_payee_prior_txn_count"],
       "first-time payees paid in 1h"),
    _f("f_newpayee_count_24h", "sequence", "24h", ["user_id", "user_to_payee_prior_txn_count"],
       "first-time payees paid in 24h"),
    _f("f_same_payee_count_30m", "sequence", "30m", ["user_id", "payee_id"], "prior payments to this payee in 30m"),
    _f("f_same_payee_amount_30m", "sequence", "30m", ["user_id", "payee_id", "amount_inr"],
       "prior INR to this payee in 30m"),
    _f("f_burst_to_new_payee", "sequence", "30m", ["user_id", "payee_id", "user_to_payee_prior_txn_count"],
       "all history with this payee is inside the last 30m (split pattern)"),
    _f("f_burst_amount", "sequence", "30m", ["user_id", "payee_id", "amount_inr"],
       "INR to this payee in the burst incl. now"),
    _f("f_collect_latency_ratio", "sequence", "all history", ["user_id", "collect_request_age_s"],
       "collect approval latency / payer's typical"),
    _f("f_collect_unknown_fast", "sequence", "event", ["collect_request_age_s", "request_source_known_contact"],
       "collect from unknown source approved within 2 min"),
    _f("f_inbound_from_new_24h", "sequence", "24h", ["user_id", "payee_id"],
       "credits from first-time senders to the payer in 24h", psp="partial"),
    _f("f_inbound_amt_24h", "sequence", "24h", ["user_id", "payee_id", "amount_inr"],
       "INR received by the payer in 24h", psp="partial"),
    # ---- payee windows -----------------------------------------------------------------------------------
    _f("f_payee_unique_payers_1h", "payee", "1h", ["payee_id", "user_id"], "distinct payers into payee in 1h",
       psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_unique_payers_7d", "payee", "7d", ["payee_id", "user_id"], "distinct payers into payee in 7d",
       psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_inbound_count_7d", "payee", "7d", ["payee_id"], "payments into payee in 7d",
       psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_new_payer_share_7d", "payee", "7d", ["payee_id", "user_to_payee_prior_txn_count"],
       "share of payee's 7d inbound from first-time payers", psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_outbound_count_24h", "payee", "24h", ["payee_id", "user_id"], "payee's own payments out in 24h",
       psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_passthrough_24h", "payee", "24h", ["payee_id", "user_id", "amount_inr"],
       "payee outbound / inbound INR in 24h", psp="partial", issuer="partial", wallet="partial"),
    _f("f_payee_activity_age_days", "payee", "all history", ["payee_id", "ts"],
       "days since payee's first observed inbound payment", psp="partial", issuer="partial", wallet="partial"),
    # ---- graph snapshots (built nightly, strictly before the row's IST day) ---------------------------
    _f("f_g_payee_degree", "graph", "snapshot", ["user_id", "payee_id"], "payee's distinct counterparties",
       psp="partial", issuer="partial", wallet="none"),
    _f("f_g_payee_dist_known_mule", "graph", "snapshot", ["user_id", "payee_id"],
       "hops to a confirmed mule (0-3, 4 = none), hubs excluded", psp="partial", issuer="partial", wallet="none"),
    _f("f_g_payee_mule_neighbors", "graph", "snapshot", ["user_id", "payee_id"],
       "confirmed mules among payee's counterparties", psp="partial", issuer="partial", wallet="none"),
    _f("f_g_payee_device_accounts", "graph", "snapshot", ["user_id", "device_id"],
       "accounts sharing a handset with the payee", psp="partial", issuer="partial", wallet="none"),
    _f("f_g_user_device_accounts", "graph", "snapshot", ["user_id", "device_id"],
       "other accounts seen on the payer's current handset", wallet="none"),
]

BY_NAME = {f.name: f for f in REGISTRY}
FEATURES = [f.name for f in REGISTRY]


def family_features(*families: str) -> list[str]:
    return [f.name for f in REGISTRY if f.family in families]


def registry_frame() -> pd.DataFrame:
    rows = []
    for f in REGISTRY:
        d = asdict(f)
        d["requires"] = ",".join(f.requires)
        rows.append(d)
    return pd.DataFrame(rows)


def graph_or_payee_features() -> list[str]:
    """Inputs of the L2 payee/mule channel."""
    names = family_features("payee", "graph")
    return names + ["f_payee_age_days", "f_payee_unique_payers_24h", "f_payee_inbound_amt_24h",
                    "f_payee_complaints", "f_payee_verified", "f_payee_first_time"]
