"""SQLite alert store: analyst queue, case dispositions, appeals, feedback and an audit log.

Identifiers are stored as salted HMACs (privacy.hmac_id); no message content is ever stored; rows
expire after the configured retention TTL (``purge_expired``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY, txn_id TEXT, event_ts INTEGER, user_hash TEXT, payee_hash TEXT, txn_type TEXT,
  amount_inr REAL, action TEXT, band TEXT, risk REAL, reasons TEXT, l0_fired TEXT, evidence TEXT,
  status TEXT DEFAULT 'open', sla_due REAL, inserted_at REAL);
CREATE INDEX IF NOT EXISTS ix_alerts_risk ON alerts(status, risk DESC);
CREATE TABLE IF NOT EXISTS cases (
  alert_id TEXT PRIMARY KEY, disposition TEXT, note TEXT, analyst TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS appeals (
  appeal_id TEXT PRIMARY KEY, alert_id TEXT, reason TEXT, created_at REAL, sla_due REAL, status TEXT);
CREATE TABLE IF NOT EXISTS feedback (
  alert_id TEXT, label TEXT, source TEXT, at REAL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, actor TEXT, action TEXT, alert_id TEXT);
"""
DISPOSITIONS = {"confirm_scam", "release", "needs_info"}
HOLD_SLA_S = 30 * 60
APPEAL_SLA_S = 24 * 3600


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.db.execute(sql, args)
            rows = cur.fetchall()
            self.db.commit()
            return rows

    def count(self) -> int:
        return int(self._q("SELECT COUNT(*) AS n FROM alerts")[0]["n"])

    def count_seeded(self) -> int:
        return int(self._q("SELECT COUNT(*) AS n FROM alerts WHERE alert_id LIKE 't-%'")[0]["n"])

    def add_alert(self, rec: dict[str, Any], now: float | None = None) -> str:
        now = time.time() if now is None else now
        aid = rec.get("alert_id") or uuid.uuid4().hex[:12]
        self._q("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            aid, rec["txn_id"], int(rec["event_ts"]), rec["user_hash"], rec["payee_hash"], rec["txn_type"],
            float(rec["amount_inr"]), rec["action"], rec["band"], float(rec["risk"]), json.dumps(rec["reasons"]),
            json.dumps(rec["l0_fired"]), json.dumps(rec.get("evidence", {})), "open",
            now + (HOLD_SLA_S if rec["action"] == "A3" else 4 * 3600), now))
        return aid

    def queue(self, status: str = "open", limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._q("""SELECT a.alert_id, a.txn_id, a.event_ts, a.user_hash, a.payee_hash, a.txn_type, a.amount_inr,
                          a.action, a.band, a.risk, a.reasons, a.l0_fired, a.status, a.sla_due, c.disposition, c.note
                          FROM alerts a LEFT JOIN cases c ON a.alert_id = c.alert_id
                          WHERE (? = 'all' OR a.status = ?)
                          ORDER BY CASE a.action WHEN 'A3' THEN 0 ELSE 1 END, a.risk DESC LIMIT ? OFFSET ?""",
                       (status, status, limit, offset))
        return [_row(r) for r in rows]

    def evidence(self, alert_id: str, actor: str) -> dict | None:
        rows = self._q("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        if not rows:
            return None
        self.audit(actor, "view_evidence", alert_id)
        return _row(rows[0], with_evidence=True)

    def disposition(self, alert_id: str, disposition: str, note: str, analyst: str) -> bool:
        if disposition not in DISPOSITIONS:
            raise ValueError(f"disposition must be one of {sorted(DISPOSITIONS)}")
        if not self._q("SELECT 1 FROM alerts WHERE alert_id = ?", (alert_id,)):
            return False
        now = time.time()
        self._q("INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?)", (alert_id, disposition, note[:500], analyst, now))
        status = {"confirm_scam": "confirmed_scam", "release": "released", "needs_info": "needs_info"}[disposition]
        self._q("UPDATE alerts SET status = ? WHERE alert_id = ?", (status, alert_id))
        label = {"confirm_scam": "scam", "release": "legit", "needs_info": "unknown"}[disposition]
        self._q("INSERT INTO feedback VALUES (?,?,?,?)", (alert_id, label, "analyst", now))
        self.audit(analyst, f"disposition:{disposition}", alert_id)
        return True

    def appeal(self, alert_id: str, reason: str, actor: str) -> dict | None:
        if not self._q("SELECT 1 FROM alerts WHERE alert_id = ?", (alert_id,)):
            return None
        now = time.time()
        aid = uuid.uuid4().hex[:12]
        self._q("INSERT INTO appeals VALUES (?,?,?,?,?,?)", (aid, alert_id, reason[:500], now, now + APPEAL_SLA_S, "open"))
        self.audit(actor, "appeal", alert_id)
        return {"appeal_id": aid, "alert_id": alert_id, "sla_due": now + APPEAL_SLA_S, "status": "open"}

    def appeals(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM appeals ORDER BY created_at DESC")]

    def audit(self, actor: str, action: str, alert_id: str | None) -> None:
        self._q("INSERT INTO audit (at, actor, action, alert_id) VALUES (?,?,?,?)", (time.time(), actor, action, alert_id))

    def audit_log(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))]

    def feedback(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM feedback ORDER BY at")]

    def purge_expired(self, now: float, ttl_days: float) -> int:
        cutoff = now - ttl_days * 86400
        ids = [r["alert_id"] for r in self._q("SELECT alert_id FROM alerts WHERE inserted_at < ?", (cutoff,))]
        for t in ("cases", "appeals", "feedback"):
            self._q(f"DELETE FROM {t} WHERE alert_id IN (SELECT alert_id FROM alerts WHERE inserted_at < ?)", (cutoff,))
        self._q("DELETE FROM alerts WHERE inserted_at < ?", (cutoff,))
        return len(ids)


def _row(r: sqlite3.Row, with_evidence: bool = False) -> dict:
    d = dict(r)
    for k in ("reasons", "l0_fired"):
        if k in d and isinstance(d[k], str):
            d[k] = json.loads(d[k])
    if "evidence" in d:
        d["evidence"] = json.loads(d["evidence"]) if with_evidence else None
        if not with_evidence:
            d.pop("evidence")
    return d
