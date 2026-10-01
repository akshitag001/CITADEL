"""The single chokepoint for time conventions.

All timestamps in Citadel are int64 Unix seconds (UTC). Every hour-of-day and day-of-week value is IST
(UTC+05:30) and is derived HERE and nowhere else, so benign traffic, attacks and features can never
disagree about what "night" means. Day of week: Monday = 0 ... Sunday = 6.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

IST_OFFSET_S = 5 * 3600 + 30 * 60
DAY_S = 86400
HOUR_S = 3600
IST = timezone(timedelta(seconds=IST_OFFSET_S))

# Relative UPI volume by IST hour. Shape: near-zero 1-5am, morning ramp, lunch and evening peaks.
DIURNAL_WEIGHTS = np.array([
    0.9, 0.45, 0.25, 0.18, 0.2, 0.4,      # 00-05
    1.0, 2.2, 3.6, 4.6, 5.4, 5.8,         # 06-11
    6.0, 5.6, 5.0, 4.8, 5.0, 5.6,         # 12-17
    6.4, 6.8, 6.6, 5.6, 3.8, 2.0,         # 18-23
])
DIURNAL = DIURNAL_WEIGHTS / DIURNAL_WEIGHTS.sum()
WEEKLY = np.array([1.0, 0.98, 0.98, 1.0, 1.05, 1.12, 0.95])  # Mon..Sun
NIGHT_HOURS = (23, 0, 1, 2, 3, 4, 5)


def start_epoch(start_date: str) -> int:
    """IST midnight of ``start_date`` (YYYY-MM-DD) as Unix seconds."""
    d = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=IST)
    return int(d.timestamp())


def hour_ist(ts: np.ndarray | int) -> np.ndarray:
    return ((np.asarray(ts, dtype=np.int64) + IST_OFFSET_S) // HOUR_S) % 24


def dow_ist(ts: np.ndarray | int) -> np.ndarray:
    # 1970-01-01 (IST) was a Thursday -> index 3 with Monday = 0.
    return (((np.asarray(ts, dtype=np.int64) + IST_OFFSET_S) // DAY_S) + 3) % 7


def ist_day_number(ts: np.ndarray | int) -> np.ndarray:
    """Absolute IST calendar day number (days since 1970-01-01 IST)."""
    return (np.asarray(ts, dtype=np.int64) + IST_OFFSET_S) // DAY_S


def day_index(ts: np.ndarray | int, t0: int) -> np.ndarray:
    """Simulation day index relative to the IST midnight ``t0``."""
    return (np.asarray(ts, dtype=np.int64) - t0) // DAY_S


def day_of_month(ts: np.ndarray | int) -> np.ndarray:
    days = ist_day_number(ts)
    return np.array([(datetime(1970, 1, 1) + timedelta(days=int(d))).day for d in np.atleast_1d(days)])


def is_night(hour: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(hour), NIGHT_HOURS)


def sample_hours(rng: np.random.Generator, n: int, preferred: np.ndarray | None = None,
                 personal_share: float = 0.5) -> np.ndarray:
    """Draw IST hours from the global diurnal curve, optionally mixed with a per-row preferred hour."""
    hours = rng.choice(24, size=n, p=DIURNAL)
    if preferred is not None:
        use_personal = rng.random(n) < personal_share
        jitter = np.rint(rng.normal(0, 1.5, n)).astype(int)
        personal = (np.asarray(preferred) + jitter) % 24
        hours = np.where(use_personal, personal, hours)
    return hours


def compose_ts(t0: int, day: np.ndarray, hour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IST (day, hour) -> Unix seconds, uniform within the hour."""
    secs = rng.integers(0, HOUR_S, size=len(day))
    return (t0 + np.asarray(day, dtype=np.int64) * DAY_S + np.asarray(hour, dtype=np.int64) * HOUR_S
            + secs).astype(np.int64)


def fmt(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), IST).strftime("%Y-%m-%d %H:%M:%S IST")
