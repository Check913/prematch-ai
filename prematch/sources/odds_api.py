# -*- coding: utf-8 -*-
"""Connecteur The Odds API (multi-bookmakers, temps reel).

Optionnel : necessite une cle gratuite (https://the-odds-api.com) placee
dans la variable d'environnement `ODDS_API_KEY`. Le plan gratuit autorise
500 requetes/mois : le scheduler ne l'interroge donc que toutes les 6 heures
par defaut et met en cache les reponses.

Apport par rapport aux CSV : plusieurs bookmakers par marche (donc la
possibilite de prendre la meilleure cote), avec les totaux et handicaps.
"""
from __future__ import annotations

import difflib
from datetime import datetime, timezone
from typing import Any

from ..features import canonical_team
from ..logging_setup import get_logger
from .football_data import match_key

LOGGER = get_logger("prematch.sources.odds_api")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY_TO_DIV = {
    "soccer_epl": "E0",
    "soccer_efl_champ": "E1",
    "soccer_spain_la_liga": "SP1",
    "soccer_germany_bundesliga": "D1",
    "soccer_italy_serie_a": "I1",
    "soccer_france_ligue_one": "F1",
    "soccer_netherlands_eredivisie": "N1",
    "soccer_portugal_primeira_liga": "P1",
    "soccer_belgium_first_div": "B1",
    "soccer_turkey_super_league": "T1",
}


def _team_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, canonical_team(left),
                                   canonical_team(right)).ratio()


def find_match_key(db, div: str, day: str, home: str, away: str,
                   threshold: float = 0.72) -> str | None:
    """Retrouve la cle d'un match deja connu a partir des noms du bookmaker."""
    candidates = [dict(row) for row in db.matches_on(day) if row["div"] == div]
    best_key, best_score = None, threshold
    for candidate in candidates:
        score = (_team_similarity(home, candidate["home"])
                 + _team_similarity(away, candidate["away"])) / 2.0
        if score > best_score:
            best_key, best_score = candidate["key"], score
    return best_key


def fetch_odds(client, cfg, sport_key: str,
               ttl: int = 6 * 3600) -> list[dict[str, Any]] | None:
    """Recupere les cotes d'un championnat (None si pas de cle)."""
    api_key = cfg.secret(str(cfg.get("sources.odds_api.api_key_env",
                                     "ODDS_API_KEY")))
    if not api_key:
        LOGGER.info("pas de cle ODDS_API_KEY : source ignoree")
        return None

    regions = cfg.get("sources.odds_api.regions", "eu")
    markets = cfg.get("sources.odds_api.markets", "h2h,totals")
    url = (f"{BASE_URL}/sports/{sport_key}/odds/?apiKey={api_key}"
           f"&regions={regions}&markets={markets}&oddsFormat=decimal")
    try:
        payload = client.get_json(url, ttl=ttl)
    except Exception as exc:
        LOGGER.warning("The Odds API (%s) : %s", sport_key, exc)
        return None
    return payload if isinstance(payload, list) else None


def sync(db, client, cfg) -> dict[str, Any]:
    """Synchronise les cotes multi-bookmakers sur les matchs connus."""
    api_cfg = cfg.section("sources").get("odds_api", {})
    if not api_cfg.get("enabled"):
        return {"skipped": True}
    if not cfg.secret(str(api_cfg.get("api_key_env", "ODDS_API_KEY"))):
        return {"skipped": True, "reason": "cle absente"}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added = matched = 0
    for sport_key in api_cfg.get("sport_keys", []):
        events = fetch_odds(client, cfg, sport_key)
        if not events:
            continue
        div = SPORT_KEY_TO_DIV.get(sport_key, sport_key)
        for event in events:
            kickoff = str(event.get("commence_time", ""))[:10]
            home = str(event.get("home_team", ""))
            away = str(event.get("away_team", ""))
            if not kickoff or not home or not away:
                continue
            key = find_match_key(db, div, kickoff, home, away)
            if not key:
                continue
            matched += 1
            for bookmaker in event.get("bookmakers", []):
                book = str(bookmaker.get("key", ""))[:24].upper()
                for market in bookmaker.get("markets", []):
                    name = market.get("key")
                    if name == "h2h":
                        for outcome in market.get("outcomes", []):
                            label = outcome.get("name", "")
                            selection = ("H" if _team_similarity(label, home) >
                                         _team_similarity(label, away) else "A")
                            if label.lower() == "draw":
                                selection = "D"
                            added += db.add_odds([{
                                "match_key": key, "book": book,
                                "market": "1X2", "selection": selection,
                                "line": None, "price": outcome.get("price"),
                                "is_closing": 0, "captured_at": now,
                                "source": "odds-api"}])
                    elif name == "totals":
                        for outcome in market.get("outcomes", []):
                            line = outcome.get("point")
                            selection = ("OVER" if str(outcome.get("name", ""))
                                         .lower().startswith("over") else "UNDER")
                            if line is None:
                                continue
                            added += db.add_odds([{
                                "match_key": key, "book": book,
                                "market": "OU", "selection": selection,
                                "line": float(line),
                                "price": outcome.get("price"),
                                "is_closing": 0, "captured_at": now,
                                "source": "odds-api"}])
    db.log_event("ingest_odds_api", f"{added} cotes, {matched} matchs apparies")
    return {"odds": added, "matches": matched}
