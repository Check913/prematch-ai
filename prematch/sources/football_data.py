# -*- coding: utf-8 -*-
"""Connecteur football-data.co.uk.

Cette source publie, en CSV libre et sans cle d'API :
  * l'historique complet des resultats par division et par saison, avec les
    cotes d'ouverture ET de fermeture de plusieurs bookmakers, les xG et les
    statistiques de match (tirs, cartons...) ;
  * un fichier `fixtures.csv` des matchs a venir, avec les cotes du moment.

C'est la colonne vertebrale du systeme : entrainement, backtest honnete
(cotes de fermeture) et detection de value en temps reel.

URL : https://www.football-data.co.uk/mmz4281/<saison>/<division>.csv
"""
from __future__ import annotations

import io
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from ..features import canonical_team
from ..logging_setup import get_logger

LOGGER = get_logger("prematch.sources.football_data")

BASE_URL = "https://www.football-data.co.uk"

# Code division -> (pays, nom lisible)
LEAGUES: dict[str, tuple[str, str]] = {
    "E0": ("Angleterre", "Premier League"),
    "E1": ("Angleterre", "Championship"),
    "E2": ("Angleterre", "League One"),
    "E3": ("Angleterre", "League Two"),
    "SC0": ("Ecosse", "Scottish Premiership"),
    "SC1": ("Ecosse", "Scottish Championship"),
    "D1": ("Allemagne", "Bundesliga"),
    "D2": ("Allemagne", "2. Bundesliga"),
    "I1": ("Italie", "Serie A"),
    "I2": ("Italie", "Serie B"),
    "SP1": ("Espagne", "LaLiga"),
    "SP2": ("Espagne", "LaLiga 2"),
    "F1": ("France", "Ligue 1"),
    "F2": ("France", "Ligue 2"),
    "N1": ("Pays-Bas", "Eredivisie"),
    "B1": ("Belgique", "Pro League"),
    "P1": ("Portugal", "Primeira Liga"),
    "T1": ("Turquie", "Super Lig"),
    "G1": ("Grece", "Super League"),
}

# Bookmakers 1N2 (ouverture / temps reel) : code -> colonnes H, D, A
BOOKS_1X2: dict[str, tuple[str, str, str]] = {
    "B365": ("B365H", "B365D", "B365A"),
    "PS": ("PSH", "PSD", "PSA"),
    "BFD": ("BFDH", "BFDD", "BFDA"),
    "BV": ("BVH", "BVD", "BVA"),
    "BW": ("BWH", "BWD", "BWA"),
    "PP": ("PPH", "PPD", "PPA"),
    "SKB": ("SKBH", "SKBD", "SKBA"),
    "BFE": ("BFEH", "BFED", "BFEA"),
    "AVG": ("AvgH", "AvgD", "AvgA"),
    "MAX": ("MaxH", "MaxD", "MaxA"),
}

# Cotes de fermeture (closing) : ce que le marche savait au coup d'envoi.
BOOKS_1X2_CLOSING: dict[str, tuple[str, str, str]] = {
    "B365": ("B365CH", "B365CD", "B365CA"),
    "PS": ("PSCH", "PSCD", "PSCA"),
    "BFD": ("BFDCH", "BFDCD", "BFDCA"),
    "BV": ("BVCH", "BVCD", "BVCA"),
    "BW": ("BWCH", "BWCD", "BWCA"),
    "PP": ("PPCH", "PPCD", "PPCA"),
    "SKB": ("SKBCH", "SKBCD", "SKBCA"),
    "BFE": ("BFECH", "BFECD", "BFECA"),
    "AVG": ("AvgCH", "AvgCD", "AvgCA"),
    "MAX": ("MaxCH", "MaxCD", "MaxCA"),
}

BOOKS_OU: dict[str, tuple[str, str]] = {
    "B365": ("B365>2.5", "B365<2.5"),
    "MAX": ("Max>2.5", "Max<2.5"),
    "AVG": ("Avg>2.5", "Avg<2.5"),
    "BFE": ("BFE>2.5", "BFE<2.5"),
}

BOOKS_AH: dict[str, tuple[str, str]] = {
    "B365": ("B365AHH", "B365AHA"),
    "MAX": ("MaxAHH", "MaxAHA"),
    "AVG": ("AvgAHH", "AvgAHA"),
    "BFE": ("BFEAHH", "BFEAHA"),
}

# Colonnes de resultat / stats
RESULT_COLUMNS = {
    "hthg": "HTHG", "htag": "HTAG",
    "hxg": "HxG", "axg": "AxG",
}


def season_code(year_start: int) -> str:
    """2026 -> '2627'."""
    return f"{year_start % 100:02d}{(year_start + 1) % 100:02d}"


def current_season(today: datetime | None = None) -> str:
    """Saison en cours au sens football europeen (juillet -> juin)."""
    today = today or datetime.now(timezone.utc)
    year = today.year if today.month >= 7 else today.year - 1
    return season_code(year)


def season_start(season: str) -> int:
    """'2627' -> 2026."""
    return 2000 + int(str(season)[:2])


def match_key(div: str, day: str, home: str, away: str) -> str:
    return "|".join([str(div), day, canonical_team(home),
                     canonical_team(away)])


def _num(value: Any) -> float | None:
    """Nombre strictement positif (cotes)."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        number = float(text)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _goal(value: Any) -> int | None:
    """Nombre de buts : 0 est une valeur valide (0-0, 2-0...)."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        if name in row:
            value = _num(row[name])
            if value is not None:
                return value
    return None


def _clean_str(value: Any) -> str:
    """Convertit une cellule CSV en chaine propre (NaN -> '')."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _parse_date(value: Any) -> str | None:
    if value is None or str(value).strip() in ("", "nan"):
        return None
    stamp = pd.to_datetime(str(value).strip(), dayfirst=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp.strftime("%Y-%m-%d")


def _parse_time(value: Any) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    match = re.match(r"^(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def _odds_rows(key: str, row: dict[str, Any], closing: bool,
               captured_at: str, source: str) -> list[dict]:
    """Extrait toutes les cotes exploitables d'une ligne CSV."""
    rows: list[dict] = []
    book_map = BOOKS_1X2_CLOSING if closing else BOOKS_1X2
    for book, (col_h, col_d, col_a) in book_map.items():
        prices = [_pick(row, [col_h]), _pick(row, [col_d]),
                  _pick(row, [col_a])]
        if any(p is None or p <= 1.01 for p in prices):
            continue
        for selection, price in zip(("H", "D", "A"), prices):
            rows.append({"match_key": key, "book": book, "market": "1X2",
                         "selection": selection, "line": None,
                         "price": float(price), "is_closing": int(closing),
                         "captured_at": captured_at, "source": source})

    if not closing:
        for book, (col_over, col_under) in BOOKS_OU.items():
            over, under = _pick(row, [col_over]), _pick(row, [col_under])
            if over and under:
                rows.append({"match_key": key, "book": book, "market": "OU",
                             "selection": "OVER", "line": 2.5,
                             "price": over, "is_closing": 0,
                             "captured_at": captured_at, "source": source})
                rows.append({"match_key": key, "book": book, "market": "OU",
                             "selection": "UNDER", "line": 2.5,
                             "price": under, "is_closing": 0,
                             "captured_at": captured_at, "source": source})

        handicap = _pick(row, ["AHh", "AHCh"])
        if handicap is not None:
            for book, (col_h, col_a) in BOOKS_AH.items():
                home_price, away_price = _pick(row, [col_h]), _pick(row, [col_a])
                if not (home_price and away_price):
                    continue
                rows.append({"match_key": key, "book": book, "market": "AH",
                             "selection": "HOME", "line": float(handicap),
                             "price": home_price, "is_closing": 0,
                             "captured_at": captured_at, "source": source})
                rows.append({"match_key": key, "book": book, "market": "AH",
                             "selection": "AWAY", "line": -float(handicap),
                             "price": away_price, "is_closing": 0,
                             "captured_at": captured_at, "source": source})
    return rows


def parse_season_csv(text: str, div: str, season: str,
                     source: str = "football-data.co.uk"
                     ) -> tuple[list[dict], list[dict]]:
    """Parse un fichier de saison complet -> (matchs, cotes)."""
    if not text or len(text) < 50:
        return [], []
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str)
    except Exception as exc:  # pragma: no cover - CSV corrompu
        LOGGER.warning("CSV illisible (%s %s) : %s", div, season, exc)
        return [], []

    matches: list[dict] = []
    odds: list[dict] = []
    for raw in frame.to_dict("records"):
        day = _parse_date(raw.get("Date"))
        home = _clean_str(raw.get("HomeTeam"))
        away = _clean_str(raw.get("AwayTeam"))
        if not day or not home or not away:
            continue
        key = match_key(div, day, home, away)
        fthg, ftag = _goal(raw.get("FTHG")), _goal(raw.get("FTAG"))
        ftr = _clean_str(raw.get("FTR")) or None
        record = {
            "key": key, "div": div, "season": season, "date": day,
            "time": _parse_time(raw.get("Time")),
            "home": home, "away": away,
            "fthg": fthg, "ftag": ftag, "ftr": ftr,
            "hthg": _goal(raw.get(RESULT_COLUMNS["hthg"])),
            "htag": _goal(raw.get(RESULT_COLUMNS["htag"])),
            "hxg": _num(raw.get(RESULT_COLUMNS["hxg"])),
            "axg": _num(raw.get(RESULT_COLUMNS["axg"])),
            "source": source,
        }
        matches.append(record)

        kickoff = f"{day}T{(record['time'] or '12:00')}:00Z"
        odds.extend(_odds_rows(key, raw, closing=False,
                               captured_at=kickoff, source=source))
        odds.extend(_odds_rows(key, raw, closing=True,
                               captured_at=kickoff, source=source))
    return matches, odds


def parse_fixtures_csv(text: str, source: str = "football-data.co.uk"
                       ) -> tuple[list[dict], list[dict]]:
    """Parse fixtures.csv (matchs a venir + cotes du moment)."""
    if not text or len(text) < 50:
        return [], []
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("fixtures.csv illisible : %s", exc)
        return [], []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    matches: list[dict] = []
    odds: list[dict] = []
    for raw in frame.to_dict("records"):
        day = _parse_date(raw.get("Date"))
        home = _clean_str(raw.get("HomeTeam"))
        away = _clean_str(raw.get("AwayTeam"))
        div = _clean_str(raw.get("Div"))
        if not day or not home or not away or not div:
            continue
        key = match_key(div, day, home, away)
        matches.append({
            "key": key, "div": div, "season": current_season(),
            "date": day, "time": _parse_time(raw.get("Time")),
            "home": home, "away": away,
            "fthg": None, "ftag": None, "ftr": None,
            "hthg": None, "htag": None, "hxg": None, "axg": None,
            "source": source,
        })
        odds.extend(_odds_rows(key, raw, closing=False, captured_at=now,
                               source=source))
    return matches, odds


# ---------------------------------------------------------------------------
# Telechargement (via le client poli) et synchronisation en base
# ---------------------------------------------------------------------------


def season_url(div: str, season: str, base_url: str = BASE_URL) -> str:
    return f"{base_url}/mmz4281/{season}/{div}.csv"


def fixtures_url(base_url: str = BASE_URL) -> str:
    return f"{base_url}/fixtures.csv"


def sync_history(db, client, cfg) -> dict[str, Any]:
    """Telecharge les saisons configurees et remplit la base."""
    source_cfg = cfg.section("sources").get("football_data", {})
    divisions = list(source_cfg.get("divisions", []))
    seasons = list(source_cfg.get("seasons", [current_season()]))
    ttl = int(source_cfg.get("history_ttl_s", 86400))
    base_url = str(source_cfg.get("base_url", BASE_URL))

    total_matches = total_odds = 0
    errors: list[str] = []
    for season in seasons:
        for div in divisions:
            url = season_url(div, season, base_url)
            try:
                text = client.get_text(url, ttl=ttl)
            except Exception as exc:
                errors.append(f"{div}/{season}: {exc}")
                LOGGER.debug("saison indisponible %s : %s", url, exc)
                continue
            matches, odds = parse_season_csv(text, div, season)
            if not matches:
                continue
            total_matches += db.upsert_matches(matches)
            total_odds += db.add_odds(odds)
            LOGGER.info("%s %s : %s matchs, %s cotes", div, season,
                        len(matches), len(odds))
    db.log_event("ingest_history",
                 f"{total_matches} matchs / {total_odds} cotes "
                 f"({len(errors)} erreurs)")
    return {"matches": total_matches, "odds": total_odds, "errors": errors}


def sync_fixtures(db, client, cfg) -> dict[str, Any]:
    """Rafraichit les matchs a venir et leurs cotes."""
    source_cfg = cfg.section("sources").get("football_data", {})
    ttl = int(source_cfg.get("fixtures_ttl_s", 900))
    base_url = str(source_cfg.get("base_url", BASE_URL))
    divisions = set(source_cfg.get("divisions", []))

    text = client.get_text(fixtures_url(base_url), ttl=ttl)
    matches, odds = parse_fixtures_csv(text)
    matches = [m for m in matches if not divisions or m["div"] in divisions]
    if not matches:
        return {"matches": 0, "odds": 0}
    keys = {m["key"] for m in matches}
    odds = [o for o in odds if o["match_key"] in keys]
    n_matches = db.upsert_matches(matches)
    n_odds = db.add_odds(odds)
    LOGGER.info("fixtures : %s matchs a venir, %s cotes", n_matches, n_odds)
    db.log_event("ingest_fixtures", f"{n_matches} matchs / {n_odds} cotes")
    return {"matches": n_matches, "odds": n_odds}


def history_frame(db, since: str | None = None) -> pd.DataFrame:
    """Charge l'historique en DataFrame pret pour les features."""
    rows = db.finished_matches(until=None)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(r) for r in rows])
    frame = frame.rename(columns={"date": "date", "time": "time"})
    frame["time"] = frame["time"].fillna("12:00")
    if since:
        frame = frame[frame["date"] >= since]
    return frame.reset_index(drop=True)


def upcoming_frame(db, days_ahead: int = 3) -> pd.DataFrame:
    """Matchs programmes dans les prochains jours."""
    today = datetime.now(timezone.utc).date()
    since = today.isoformat()
    until = (today + timedelta(days=days_ahead)).isoformat()
    rows = [dict(r) for r in db.upcoming_matches(since=since)
            if r["date"] <= until]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["time"] = frame["time"].fillna("12:00")
    return frame


def league_label(div: str) -> str:
    country, name = LEAGUES.get(div, ("", div))
    return f"{name} ({country})" if country else name
