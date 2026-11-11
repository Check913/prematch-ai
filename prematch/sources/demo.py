# -*- coding: utf-8 -*-
"""Generateur de donnees synthetiques (mode demo / tests hors ligne).

On simule un championnat ferme dont on connait les vraies forces, puis on
genere des cotes avec une marge realiste. Cela permet de faire tourner tout
le pipeline (features, modeles, value, Telegram) sans reseau ni cle d'API.
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

import numpy as np

from ..markets import matrix_markets, poisson_matrix
from .football_data import match_key

TEAM_NAMES = [
    "Paris SG", "Marseille", "Monaco", "Lille", "Lyon", "Nice", "Lens",
    "Rennes", "Lens B", "Strasbourg", "Brest", "Toulouse", "Nantes",
    "Montpellier", "Reims", "Angers", "Le Havre", "Auxerre", "Metz",
    "Lorient",
]


def _prices_from_probs(probs: list[float], margin: float,
                       rng: random.Random) -> list[float]:
    """Cotes = proba vraie degradee par la marge + bruit de marche."""
    prices = []
    for p in probs:
        fair = 1.0 / max(p, 1e-3)
        noisy = fair * (1.0 - margin) * rng.uniform(0.97, 1.03)
        prices.append(round(max(1.02, noisy), 2))
    return prices


def generate(n_teams: int = 18, seasons: int = 3, seed: int = 7
             ) -> tuple[list[dict], list[dict]]:
    """Simule `seasons` saisons completes et retourne (matchs, cotes)."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    teams = TEAM_NAMES[:n_teams]
    attack = dict(zip(teams, rng.normal(0, 0.28, len(teams))))
    defence = dict(zip(teams, rng.normal(0, 0.24, len(teams))))
    home_adv = 0.22
    base = 1.32

    start_year = 2023
    matches: list[dict] = []
    odds: list[dict] = []

    for season_index in range(seasons):
        season = f"{start_year + season_index}{str(start_year + season_index + 1)[-2:]}"
        season_label = season[2:]
        start = datetime(start_year + season_index, 8, 10)
        day = start
        fixtures = []
        for home in teams:
            for away in teams:
                if home != away:
                    fixtures.append((home, away))
        py_rng.shuffle(fixtures)

        for index, (home, away) in enumerate(fixtures):
            if index % (len(teams) - 1) == 0:
                day += timedelta(days=7)
            lam_h = base * np.exp(attack[home] + defence[away] + home_adv / 2)
            lam_a = base * np.exp(attack[away] + defence[home] - home_adv / 2)
            hg = int(rng.poisson(lam_h))
            ag = int(rng.poisson(lam_a))

            date_str = day.strftime("%Y-%m-%d")
            key = match_key(season_label, date_str, home, away)
            matches.append({
                "key": key, "div": season_label, "season": season,
                "date": date_str, "time": "20:00",
                "home": home, "away": away,
                "fthg": hg, "ftag": ag,
                "ftr": "H" if hg > ag else ("A" if ag > hg else "D"),
                "hthg": max(0, hg - 1), "htag": max(0, ag - 1),
                "hxg": round(float(lam_h), 2), "axg": round(float(lam_a), 2),
                "source": "demo",
            })

            matrix = poisson_matrix(lam_h, lam_a, -0.05, 10)
            market = matrix_markets(matrix)
            probs_1x2 = [market[("1X2", "H", None)],
                         market[("1X2", "D", None)],
                         market[("1X2", "A", None)]]
            base_prices = _prices_from_probs(probs_1x2, 0.06, py_rng)
            alt_prices = _prices_from_probs(probs_1x2, 0.05, py_rng)
            for book, prices in (("DEMO", base_prices), ("ALT", alt_prices)):
                for selection, price in zip(("H", "D", "A"), prices):
                    odds.append({"match_key": key, "book": book,
                                 "market": "1X2", "selection": selection,
                                 "line": None, "price": price,
                                 "is_closing": 1,
                                 "captured_at": f"{date_str}T20:00:00Z",
                                 "source": "demo"})

            p_over = market[("OU", "OVER", 2.5)]
            prices_ou = _prices_from_probs([p_over, 1 - p_over], 0.045, py_rng)
            for selection, price in zip(("OVER", "UNDER"), prices_ou):
                odds.append({"match_key": key, "book": "DEMO", "market": "OU",
                             "selection": selection, "line": 2.5,
                             "price": price, "is_closing": 1,
                             "captured_at": f"{date_str}T20:00:00Z",
                             "source": "demo"})
    return matches, odds


def generate_upcoming(matches: list[dict], odds: list[dict],
                      n_teams: int = 18, seed: int = 7) -> None:
    """Ajoute des matchs a venir (aujourd'hui + 2 jours) pour la demo."""
    rng = np.random.default_rng(seed + 1)
    teams = TEAM_NAMES[:n_teams]
    today = datetime.now(timezone.utc).date()
    py_rng = random.Random(seed)
    for offset in (0, 1, 2):
        day = today + timedelta(days=offset)
        for index in range(0, len(teams) - 1, 2):
            home, away = teams[index], teams[index + 1]
            date_str = day.strftime("%Y-%m-%d")
            key = match_key("D0", date_str, home, away)
            lam_h = 1.32 * np.exp(rng.normal(0, 0.28))
            lam_a = 1.32 * np.exp(rng.normal(0, 0.24))
            matrix = poisson_matrix(lam_h, lam_a, -0.05, 10)
            market = matrix_markets(matrix)
            matches.append({
                "key": key, "div": "D0", "season": "2627",
                "date": date_str, "time": "20:00",
                "home": home, "away": away,
                "fthg": None, "ftag": None, "ftr": None,
                "hthg": None, "htag": None,
                "hxg": round(float(lam_h), 2),
                "axg": round(float(lam_a), 2),
                "source": "demo",
            })
            probs_1x2 = [market[("1X2", "H", None)],
                         market[("1X2", "D", None)],
                         market[("1X2", "A", None)]]
            prices = _prices_from_probs(probs_1x2, 0.06, py_rng)
            for selection, price in zip(("H", "D", "A"), prices):
                odds.append({"match_key": key, "book": "DEMO",
                             "market": "1X2", "selection": selection,
                             "line": None, "price": price, "is_closing": 0,
                             "captured_at": f"{date_str}T20:00:00Z",
                             "source": "demo"})
            p_over = market[("OU", "OVER", 2.5)]
            prices_ou = _prices_from_probs([p_over, 1 - p_over], 0.045, py_rng)
            for selection, price in zip(("OVER", "UNDER"), prices_ou):
                odds.append({"match_key": key, "book": "DEMO", "market": "OU",
                             "selection": selection, "line": 2.5,
                             "price": price, "is_closing": 0,
                             "captured_at": f"{date_str}T20:00:00Z",
                             "source": "demo"})


def sync_demo(db, cfg) -> dict[str, int]:
    """Ecrit le jeu synthetique en base (idempotent)."""
    demo_cfg = cfg.section("sources").get("demo", {})
    n_teams = int(demo_cfg.get("matches", 18))
    n_teams = 18 if n_teams > 20 else max(8, n_teams)
    matches, odds = generate(n_teams=min(n_teams, 20), seasons=3,
                             seed=int(demo_cfg.get("seed", 7)))
    generate_upcoming(matches, odds, n_teams=min(n_teams, 20),
                      seed=int(demo_cfg.get("seed", 7)))
    n_matches = db.upsert_matches(matches)
    n_odds = db.add_odds(odds)
    db.log_event("ingest_demo", f"{n_matches} matchs / {n_odds} cotes")
    return {"matches": n_matches, "odds": n_odds}
