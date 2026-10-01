# Scam sequence grammar

<!-- generated from configs/scam_library.yaml by `python -m citadel scam-doc` -->

Every scam is a four-step sequence: **S1 lure** (unobserved) -> **S2 pressure / remote access** -> **S3 collect or pay a new payee (Citadel acts here)** -> **S4 mule fan-out**. Signals fire probabilistically; unobservable signals stay missing.

| Variant | S3 type | Payments | Amount | First-time payee | S2 signals | S3 signals | Segments |
|---|---|---|---|---|---|---|---|
| **V1** Fake-refund collect request | collect | [1, 2] | [0.8, 5.0]x usual | 0.93 | call_in_progress: p=0.65<br>screen_share_active: p=0.12<br>remote_access_app_detected: p=0.06 | collect_request_age_s: p=1.0 (lognormal [75, 0.8])<br>session_duration_s: p=0.7 (lognormal [150, 0.6])<br>attempts_in_session: p=0.4 (int_range [2, 3]) | senior, low_literacy, first_time_user, young_adult, mainstream |
| **V2** Screen-share "bank support" scam | pay | [1, 3] | [0.3, 0.9] of balance | 0.95 | remote_access_app_detected: p=0.7<br>screen_share_active: p=0.72<br>call_in_progress: p=0.85 | session_duration_s: p=0.85 (lognormal [900, 0.5])<br>attempts_in_session: p=0.6 (int_range [2, 4]) | senior, low_literacy, first_time_user, mainstream |
| **V3** Fake-KYC link, then payment | pay | [1, 2] | [2.0, 8.0]x usual | 0.95 | new_device_flag: p=0.4<br>sim_change_7d: p=0.12 | session_duration_s: p=0.6 (lognormal [240, 0.6]) | senior, low_literacy, first_time_user, mainstream |
| **V4** QR "scan to receive money" | qr_pay | [1, 3] | [1.0, 6.0]x usual | 0.97 | call_in_progress: p=0.5 | session_duration_s: p=0.6 (lognormal [110, 0.5])<br>attempts_in_session: p=0.6 (int_range [2, 3]) | young_adult, first_time_user, mainstream |
| **V5** Utility / courier disconnection threat | pay | [1, 3] | INR [800, 6000] | 0.9 | call_in_progress: p=0.85 | session_duration_s: p=0.6 (lognormal [200, 0.6]) | senior, low_literacy, mainstream |
| **V6** Task / job scam | pay | [1, 3] | [3.0, 15.0]x usual | 0.85 | call_in_progress: p=0.1 | session_duration_s: p=0.4 (lognormal [180, 0.6]) | young_adult, first_time_user, mainstream |

## Columns each step can influence

| Variant | S2 | S3 | S4 |
|---|---|---|---|
| V1 | call_in_progress, remote_access_app_detected, screen_share_active | amount_inr, attempts_in_session, collect_request_age_s, payee_id, payee_name_similarity_to_known_contact, session_duration_s, txn_type | payee_inbound_unique_payers_24h, payee_inbound_amount_24h, (mule outbound rows) |
| V2 | call_in_progress, remote_access_app_detected, screen_share_active | amount_inr, attempts_in_session, payee_id, payee_name_similarity_to_known_contact, session_duration_s, txn_type | payee_inbound_unique_payers_24h, payee_inbound_amount_24h, (mule outbound rows) |
| V3 | amount_inr, new_device_flag, payee_id, sim_change_7d | amount_inr, payee_id, payee_name_similarity_to_known_contact, session_duration_s, txn_type | payee_inbound_unique_payers_24h, payee_inbound_amount_24h, (mule outbound rows) |
| V4 | call_in_progress | amount_inr, attempts_in_session, payee_id, payee_name_similarity_to_known_contact, session_duration_s, txn_type | payee_inbound_unique_payers_24h, payee_inbound_amount_24h, (mule outbound rows) |
| V5 | amount_inr, call_in_progress, payee_id | amount_inr, payee_id, payee_name_similarity_to_known_contact, session_duration_s, txn_type | payee_inbound_unique_payers_24h, payee_inbound_amount_24h, (mule outbound rows) |
| V6 | amount_inr, call_in_progress, payee_id | amount_inr, payee_id, payee_name_similarity_to_known_contact, session_duration_s, txn_type | payee_inbound_unique_payers_24h, payee_inbound_amount_24h, (mule outbound rows) |

## Evasions (attacker-controllable levers only)

| Evasion | Levers | Requires |
|---|---|---|
| Amount splitting | amount, pacing | — |
| Slow-drip pacing | pacing | — |
| Aged / recruited mule | mule_age | — |
| Payee-name mimicry | payee_naming | the QR flow shows the QR's registered name, not a saved contact, so contact-name mimicry has no lever |
| Time-of-day mimicry | time_of_day | — |
| Signal suppression (video call instead of screen share) | signal_suppression | the variant uses no screen-share or remote-access signal, so there is nothing to suppress |

## Composition space

138 legal of 210 (variant x evasion x segment). Every illegal combination is explained by the constraint it fails, e.g.:

- V2 + none + young_adult: segment 'young_adult' is not targeted by V2 (Screen-share "bank support" scam); allowed: senior, low_literacy, first_time_user, mainstream
- V3 + none + young_adult: segment 'young_adult' is not targeted by V3 (Fake-KYC link, then payment); allowed: senior, low_literacy, first_time_user, mainstream
- V3 + signal_suppression + senior: evasion 'signal_suppression' is illegal for V3: the variant uses no screen-share or remote-access signal, so there is nothing to suppress
- V4 + none + senior: segment 'senior' is not targeted by V4 (QR "scan to receive money"); allowed: young_adult, first_time_user, mainstream
- V4 + none + low_literacy: segment 'low_literacy' is not targeted by V4 (QR "scan to receive money"); allowed: young_adult, first_time_user, mainstream
- V4 + payee_name_mimicry + first_time_user: evasion 'payee_name_mimicry' is illegal for V4: the QR flow shows the QR's registered name, not a saved contact, so contact-name mimicry has no lever
- V4 + signal_suppression + first_time_user: evasion 'signal_suppression' is illegal for V4: the variant uses no screen-share or remote-access signal, so there is nothing to suppress
- V5 + none + first_time_user: segment 'first_time_user' is not targeted by V5 (Utility / courier disconnection threat); allowed: senior, low_literacy, mainstream
- V5 + none + young_adult: segment 'young_adult' is not targeted by V5 (Utility / courier disconnection threat); allowed: senior, low_literacy, mainstream
- V5 + signal_suppression + senior: evasion 'signal_suppression' is illegal for V5: the variant uses no screen-share or remote-access signal, so there is nothing to suppress
- V6 + none + senior: segment 'senior' is not targeted by V6 (Task / job scam); allowed: young_adult, first_time_user, mainstream
- V6 + none + low_literacy: segment 'low_literacy' is not targeted by V6 (Task / job scam); allowed: young_adult, first_time_user, mainstream
- V6 + signal_suppression + first_time_user: evasion 'signal_suppression' is illegal for V6: the variant uses no screen-share or remote-access signal, so there is nothing to suppress
