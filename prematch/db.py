# -*- coding: utf-8 -*-
"""Persistance SQLite : matchs, cotes, predictions, paris, modeles.

Le fichier de base vit dans `<data_dir>/prematch.db`. On utilise WAL pour
permettre la lecture pendant que le scheduler ecrit.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS matches (
    key          TEXT PRIMARY KEY,          -- div|date|home|away
    div          TEXT NOT NULL,
    season       TEXT,
    date         TEXT NOT NULL,             -- YYYY-MM-DD
    time         TEXT,                      -- HH:MM
    home         TEXT NOT NULL,
    away         TEXT NOT NULL,
    fthg         INTEGER,
    ftag         INTEGER,
    ftr          TEXT,
    hthg         INTEGER,
    htag         INTEGER,
    hxg          REAL,
    axg          REAL,
    status       TEXT DEFAULT 'scheduled',  -- scheduled | finished
    source       TEXT,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);

CREATE TABLE IF NOT EXISTS odds (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_key   TEXT NOT NULL,
    book        TEXT NOT NULL,
    market      TEXT NOT NULL,              -- 1X2 | OU | AH | BTTS
    selection   TEXT NOT NULL,              -- H | D | A | OVER | UNDER ...
    line        REAL,                       -- 2.5 pour OU, -0.5 pour AH
    price       REAL NOT NULL,
    is_closing  INTEGER NOT NULL DEFAULT 0,
    captured_at TEXT NOT NULL,
    source      TEXT,
    UNIQUE(match_key, book, market, selection, line, is_closing, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_key);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    match_key     TEXT NOT NULL,
    kickoff       TEXT,
    div           TEXT,
    home          TEXT,
    away          TEXT,
    model_version TEXT,
    market        TEXT NOT NULL,
    selection     TEXT NOT NULL,
    line          REAL,
    p_model       REAL NOT NULL,
    p_market      REAL,
    price         REAL NOT NULL,
    book          TEXT,
    edge          REAL NOT NULL,
    kelly         REAL,
    stake         REAL,
    confidence    REAL,
    status        TEXT DEFAULT 'pending',   -- pending | sent | placed | skipped
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE(match_key, market, selection, line, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_kickoff ON predictions(kickoff);

CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER UNIQUE REFERENCES predictions(id),
    match_key     TEXT,
    market        TEXT,
    selection     TEXT,
    line          REAL,
    price         REAL NOT NULL,
    stake         REAL NOT NULL,
    status        TEXT DEFAULT 'open',      -- open | won | lost | void | half
    pnl           REAL,
    clv           REAL,
    created_at    TEXT NOT NULL,
    settled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);

CREATE TABLE IF NOT EXISTS model_versions (
    version     TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    kind        TEXT,
    params      TEXT,
    metrics     TEXT,
    path        TEXT,
    is_active   INTEGER NOT NULL DEFAULT 0,
    trained_on  TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    level   TEXT,
    kind    TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_line(line: Any) -> float | None:
    return None if line is None or line == "" else float(line)


class Database:
    """Petit wrapper autour de sqlite3 (thread-safe via verrou)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                    timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._repair_status()
        self.conn.commit()

    def _repair_status(self) -> None:
        """Repaire les statuts incoherents (imports croises histor/fixtures)."""
        self.conn.execute(
            "UPDATE matches SET status='finished' "
            "WHERE fthg IS NOT NULL AND status<>'finished'")
        self.conn.execute(
            "UPDATE matches SET status='past_no_result' "
            "WHERE fthg IS NULL AND date < date('now') AND status='scheduled'")

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ matchs
    def upsert_matches(self, rows: Iterable[dict]) -> int:
        """Insere/met a jour des matchs ; ignore les champs absents."""
        count = 0
        now = utc_now()
        with self._lock, self.conn:
            for row in rows:
                if not row.get("key"):
                    continue
                if row.get("fthg") is not None:
                    status = "finished"
                else:
                    today = datetime.now(timezone.utc).date().isoformat()
                    status = str(row.get("status") or (
                        "scheduled" if row.get("date", "") >= today
                        else "past_no_result"))
                self.conn.execute(
                    """
                    INSERT INTO matches (key, div, season, date, time, home,
                        away, fthg, ftag, ftr, hthg, htag, hxg, axg, status,
                        source, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET
                        season=COALESCE(excluded.season, matches.season),
                        time=COALESCE(excluded.time, matches.time),
                        fthg=COALESCE(excluded.fthg, matches.fthg),
                        ftag=COALESCE(excluded.ftag, matches.ftag),
                        ftr=COALESCE(excluded.ftr, matches.ftr),
                        hthg=COALESCE(excluded.hthg, matches.hthg),
                        htag=COALESCE(excluded.htag, matches.htag),
                        hxg=COALESCE(excluded.hxg, matches.hxg),
                        axg=COALESCE(excluded.axg, matches.axg),
                        status=excluded.status,
                        source=COALESCE(excluded.source, matches.source),
                        updated_at=excluded.updated_at
                    """,
                    (row["key"], row.get("div") or "", row.get("season"),
                     row["date"], row.get("time"), row["home"], row["away"],
                     row.get("fthg"), row.get("ftag"), row.get("ftr"),
                     row.get("hthg"), row.get("htag"), row.get("hxg"),
                     row.get("axg"), status, row.get("source"), now))
                count += 1
        return count

    def get_match(self, key: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self.conn.execute("SELECT * FROM matches WHERE key=?", (key,))
            return cur.fetchone()

    def finished_matches(self, until: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM matches WHERE status='finished' AND fthg IS NOT NULL"
        params: list[Any] = []
        if until:
            sql += " AND date <= ?"
            params.append(until)
        sql += " ORDER BY date, time"
        with self._lock:
            return list(self.conn.execute(sql, params))

    def upcoming_matches(self, since: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM matches WHERE status='scheduled' AND date >= ? "
                "ORDER BY date, time", (since,)))

    def matches_on(self, date: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM matches WHERE date=? ORDER BY time", (date,)))

    def count_matches(self) -> int:
        with self._lock:
            return int(self.conn.execute(
                "SELECT COUNT(*) FROM matches").fetchone()[0])

    # ------------------------------------------------------------------ cotes
    def add_odds(self, rows: Iterable[dict]) -> int:
        now = utc_now()
        count = 0
        with self._lock, self.conn:
            for row in rows:
                if not row.get("price") or float(row["price"]) <= 1.0:
                    continue
                try:
                    self.conn.execute(
                        """INSERT OR IGNORE INTO odds (match_key, book, market,
                           selection, line, price, is_closing, captured_at,
                           source) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (row["match_key"], row["book"], row["market"],
                         row["selection"], _norm_line(row.get("line")),
                         float(row["price"]), int(bool(row.get("is_closing"))),
                         row.get("captured_at") or now, row.get("source")))
                    count += 1
                except sqlite3.Error:
                    continue
        return count

    def latest_odds(self, match_key: str) -> list[sqlite3.Row]:
        """Derniere cote connue par (book, market, selection, line)."""
        with self._lock:
            return list(self.conn.execute(
                """
                SELECT o.* FROM odds o
                JOIN (SELECT book, market, selection, line, MAX(captured_at) AS ts
                      FROM odds WHERE match_key=? GROUP BY book, market,
                      selection, line) m
                  ON m.book=o.book AND m.market=o.market
                 AND m.selection=o.selection
                 AND (m.line IS o.line) AND m.ts=o.captured_at
                WHERE o.match_key=?
                """, (match_key, match_key)))

    def closing_odds(self, match_key: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM odds WHERE match_key=? AND is_closing=1",
                (match_key,)))

    # ------------------------------------------------------------- predictions
    def save_predictions(self, rows: Iterable[dict]) -> list[int]:
        ids: list[int] = []
        now = utc_now()
        with self._lock, self.conn:
            for row in rows:
                cur = self.conn.execute(
                    """
                    INSERT INTO predictions (match_key, kickoff, div, home, away,
                        model_version, market, selection, line, p_model, p_market,
                        price, book, edge, kelly, stake, confidence, status,
                        created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)
                    ON CONFLICT(match_key, market, selection, line, model_version)
                    DO UPDATE SET p_model=excluded.p_model,
                        p_market=excluded.p_market, price=excluded.price,
                        book=excluded.book, edge=excluded.edge,
                        kelly=excluded.kelly, stake=excluded.stake,
                        confidence=excluded.confidence,
                        updated_at=excluded.updated_at
                    """,
                    (row["match_key"], row.get("kickoff"), row.get("div"),
                     row.get("home"), row.get("away"),
                     row.get("model_version"), row["market"], row["selection"],
                     _norm_line(row.get("line")), float(row["p_model"]),
                     row.get("p_market"), float(row["price"]),
                     row.get("book"), float(row["edge"]), row.get("kelly"),
                     row.get("stake"), row.get("confidence"), now, now))
                if cur.lastrowid:
                    ids.append(int(cur.lastrowid))
        return ids

    def unsent_predictions(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM predictions WHERE status='pending' "
                "AND kickoff >= datetime('now','-15 minutes') "
                "ORDER BY edge DESC LIMIT ?", (limit,)))

    def predictions_for_day(self, date: str,
                            min_edge: float = 0.0) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM predictions WHERE substr(kickoff,1,10)=? "
                "AND edge>=? ORDER BY edge DESC, kickoff", (date, min_edge)))

    def mark_predictions(self, ids: Iterable[int], status: str) -> None:
        ids = [int(i) for i in ids]
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        with self._lock, self.conn:
            self.conn.execute(
                f"UPDATE predictions SET status=?, updated_at=? WHERE id IN ({marks})",
                [status, utc_now(), *ids])

    # ------------------------------------------------------------------ paris
    def place_bet(self, prediction_id: int, stake: float, price: float,
                  status: str = "open") -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO bets (prediction_id, match_key, market,
                   selection, line, price, stake, status, created_at)
                   SELECT id, match_key, market, selection, line, ?, ?, ?, ?
                   FROM predictions WHERE id=?""",
                (price, stake, status, utc_now(), prediction_id))
            self.conn.execute(
                "UPDATE predictions SET status='placed' WHERE id=?",
                (prediction_id,))
            return int(cur.lastrowid or 0)

    def settle_open_bets(self) -> int:
        """Solde les paris dont le match est termine (1X2/OU/BTTS/AH)."""
        settled = 0
        with self._lock:
            open_bets = list(self.conn.execute(
                """SELECT b.*, m.fthg, m.ftag FROM bets b
                   JOIN matches m ON m.key=b.match_key
                   WHERE b.status='open' AND m.status='finished'
                     AND m.fthg IS NOT NULL"""))
        for bet in open_bets:
            outcome = settle_bet(bet["market"], bet["selection"], bet["line"],
                                int(bet["fthg"]), int(bet["ftag"]))
            pnl = None
            status = "void"
            if outcome == "won":
                pnl = round(float(bet["stake"]) * (float(bet["price"]) - 1), 2)
                status = "won"
            elif outcome == "lost":
                pnl = -round(float(bet["stake"]), 2)
                status = "lost"
            with self._lock, self.conn:
                self.conn.execute(
                    "UPDATE bets SET status=?, pnl=?, settled_at=? WHERE id=?",
                    (status, pnl, utc_now(), bet["id"]))
            settled += 1
        return settled

    def bet_stats(self, days: int = 120) -> dict[str, Any]:
        with self._lock:
            rows = list(self.conn.execute(
                """SELECT status, COUNT(*) n, SUM(stake) stake, SUM(pnl) pnl
                   FROM bets WHERE settled_at >= datetime('now', ?)
                   GROUP BY status""", (f"-{int(days)} days",)))
            total = self.conn.execute(
                """SELECT COUNT(*) n, COALESCE(SUM(stake),0) stake,
                          COALESCE(SUM(pnl),0) pnl FROM bets
                   WHERE settled_at >= datetime('now', ?)""",
                (f"-{int(days)} days",)).fetchone()
        stats: dict[str, Any] = {"by_status": {r["status"]: dict(r)
                                               for r in rows}}
        n = int(total["n"] or 0)
        stake = float(total["stake"] or 0)
        pnl = float(total["pnl"] or 0)
        stats.update({"n": n, "stake": round(stake, 2),
                      "pnl": round(pnl, 2),
                      "roi": round(pnl / stake * 100, 2) if stake else 0.0})
        return stats

    # ---------------------------------------------------------------- modeles
    def register_model(self, version: str, kind: str, params: dict,
                       metrics: dict, path: str, trained_on: str | None = None,
                       activate: bool = False) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO model_versions (version, created_at,
                   kind, params, metrics, path, is_active, trained_on)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (version, utc_now(), kind, json.dumps(params, default=str),
                 json.dumps(metrics, default=str), path,
                 1 if activate else 0, trained_on))
            if activate:
                self.conn.execute(
                    "UPDATE model_versions SET is_active=0 WHERE version<>?",
                    (version,))

    def activate_model(self, version: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE model_versions SET is_active=0")
            self.conn.execute(
                "UPDATE model_versions SET is_active=1 WHERE version=?",
                (version,))

    def active_model(self) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM model_versions WHERE is_active=1 "
                "ORDER BY created_at DESC LIMIT 1").fetchone()

    def model_versions(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?",
                (limit,)))

    # ------------------------------------------------------------------- meta
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO meta (k,v) VALUES (?,?) ON CONFLICT(k) "
                "DO UPDATE SET v=excluded.v", (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self.conn.execute("SELECT v FROM meta WHERE k=?",
                                    (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["v"])
        except ValueError:
            return default

    def log_event(self, kind: str, message: str, level: str = "INFO") -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO events (ts, level, kind, message) VALUES (?,?,?,?)",
                (utc_now(), level, kind, message[:2000]))

    def recent_events(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)))


def settle_bet(market: str, selection: str, line: float | None,
               home_goals: int, away_goals: int) -> str:
    """Retourne 'won' | 'lost' | 'void' pour un marche donne."""
    market = (market or "").upper()
    selection = (selection or "").upper()
    total = home_goals + away_goals
    if market in ("1X2", "MATCH_RESULT"):
        result = "H" if home_goals > away_goals else (
            "A" if away_goals > home_goals else "D")
        return "won" if selection == result else "lost"
    if market in ("OU", "OVER_UNDER", "TOTALS"):
        if line is None:
            return "void"
        pushed = total == float(line)
        if pushed:
            return "void"
        over = total > float(line)
        return "won" if (selection.startswith("OVER") and over) or (
            selection.startswith("UNDER") and not over) else "lost"
    if market in ("BTTS", "BOTH_TEAMS_TO_SCORE"):
        btts = home_goals > 0 and away_goals > 0
        return "won" if (selection == "YES") == btts else "lost"
    if market in ("AH", "ASIAN_HANDICAP"):
        if line is None:
            return "void"
        margin = home_goals - away_goals
        if selection in ("HOME", "H"):
            value = margin + float(line)
        else:
            value = -margin + float(line)
        if abs(value) < 1e-9:
            return "void"
        return "won" if value > 0 else "lost"
    if market in ("DC", "DOUBLE_CHANCE"):
        result = "H" if home_goals > away_goals else (
            "A" if away_goals > home_goals else "D")
        combos = {"1X": {"H", "D"}, "12": {"H", "A"}, "X2": {"D", "A"},
                  "HD": {"H", "D"}, "HA": {"H", "A"}, "DA": {"D", "A"}}
        return "won" if result in combos.get(selection, set()) else "lost"
    return "void"
