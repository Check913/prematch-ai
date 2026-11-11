# -*- coding: utf-8 -*-
"""Construction des variables explicatives (sans fuite temporelle).

Chaque match est decrit uniquement avec l'information disponible AVANT le
coup d'envoi :
  * Elo (mise a jour sequentielle, avantage du terrain) ;
  * forme recente : points, buts, xG, xGA sur les 5 et 10 derniers matchs ;
  * repos entre matchs, experience (nombre de matchs joues) ;
  * confrontations directes (3 dernieres) ;
  * moyenne de buts de la division (niveau de jeu).

Les cotes ne sont JAMAIS des variables du modele : elles servent seulement
de reference de marche pour calculer l'edge. Sinon le modele apprendrait
simplement a recopier le bookmaker.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff",
    "ppg5_home", "ppg5_away", "ppg10_home", "ppg10_away",
    "gf5_home", "ga5_home", "gf5_away", "ga_away_5",
    "xg5_home", "xga5_home", "xg5_away", "xga5_away",
    "form_diff", "goals_exp_home", "goals_exp_away",
    "rest_home", "rest_away", "played_home", "played_away",
    "h2h_diff", "league_goals", "home_new", "away_new",
]

ALIASES = {
    "man united": "man utd", "manchester united": "man utd",
    "man city": "man city", "manchester city": "man city",
    "newcastle": "newcastle", "nottm forest": "nott'm forest",
    "sheffield weds": "sheffield wednesday", "qpr": "qpr",
    "paris sg": "paris saint germain", "psg": "paris saint germain",
    "bayern munich": "bayern munich", "inter": "inter",
    "spain": "spain",
}


def canonical_team(name: str) -> str:
    """Nom d'equipe normalise (minuscules, sans accents ni ponctuation)."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name)).encode(
        "ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def _to_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return date(1970, 1, 1)


@dataclass
class TeamState:
    """Etat accumule d'une equipe (uniquement du passe)."""

    elo: float = 1500.0
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_date: date | None = None
    played: int = 0

    def add(self, day: date, gf: int, ga: int, xg: float | None,
            xga: float | None) -> None:
        points = 3 if gf > ga else (1 if gf == ga else 0)
        self.history.append({"date": day, "gf": gf, "ga": ga, "pts": points,
                             "xg": xg, "xga": xga})
        self.last_date = day
        self.played += 1

    def last_n(self, n: int) -> list[dict]:
        items = list(self.history)
        return items[-n:] if n else []


def _avg(values: Iterable[float | None], default: float) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean)) if clean else default


class FeatureBuilder:
    """Construit le jeu d'entrainement et les variables des matchs a venir."""

    def __init__(self, elo_k: float = 20.0, home_elo_bonus: float = 60.0):
        self.elo_k = elo_k
        self.home_elo_bonus = home_elo_bonus
        self.teams: dict[str, TeamState] = defaultdict(TeamState)
        self.h2h: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=3))
        self.league_goals: dict[str, float] = defaultdict(lambda: 2.6)

    # ------------------------------------------------------------------ Elo
    def _expected(self, elo_home: float, elo_away: float) -> float:
        diff = elo_home + self.home_elo_bonus - elo_away
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _update_elo(self, home: str, away: str, gf: int, ga: int) -> None:
        state_h = self.teams[home]
        state_a = self.teams[away]
        expected = self._expected(state_h.elo, state_a.elo)
        score = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)
        goal_diff = abs(gf - ga)
        multiplier = 1.0
        if goal_diff == 2:
            multiplier = 1.5
        elif goal_diff >= 3:
            multiplier = (11.0 + goal_diff) / 8.0
        delta = self.elo_k * multiplier * (score - expected)
        state_h.elo += delta
        state_a.elo -= delta

    # ------------------------------------------------------------- features
    def _row_features(self, home: str, away: str, div: str,
                      day: date) -> dict[str, float]:
        sh = self.teams.get(home, TeamState())
        sa = self.teams.get(away, TeamState())
        hist_h5, hist_a5 = sh.last_n(5), sa.last_n(5)
        hist_h10, hist_a10 = sh.last_n(10), sa.last_n(10)

        ppg5_h = _avg([m["pts"] for m in hist_h5], 1.35)
        ppg5_a = _avg([m["pts"] for m in hist_a5], 1.15)
        ppg10_h = _avg([m["pts"] for m in hist_h10], 1.35)
        ppg10_a = _avg([m["pts"] for m in hist_a10], 1.15)

        gf5_h = _avg([m["gf"] for m in hist_h5], 1.35)
        ga5_h = _avg([m["ga"] for m in hist_h5], 1.25)
        gf5_a = _avg([m["gf"] for m in hist_a5], 1.15)
        ga5_a = _avg([m["ga"] for m in hist_a5], 1.45)

        xg5_h = _avg([m["xg"] for m in hist_h5], gf5_h)
        xga5_h = _avg([m["xga"] for m in hist_h5], ga5_h)
        xg5_a = _avg([m["xg"] for m in hist_a5], gf5_a)
        xga5_a = _avg([m["xga"] for m in hist_a5], ga5_a)

        rest_h = (day - sh.last_date).days if sh.last_date else 14
        rest_a = (day - sa.last_date).days if sa.last_date else 14

        h2h = self.h2h.get((home, away), deque())
        h2h_diff = _avg([m["gf"] - m["ga"] for m in h2h], 0.0)

        return {
            "elo_home": sh.elo, "elo_away": sa.elo,
            "elo_diff": sh.elo - sa.elo,
            "ppg5_home": ppg5_h, "ppg5_away": ppg5_a,
            "ppg10_home": ppg10_h, "ppg10_away": ppg10_a,
            "gf5_home": gf5_h, "ga5_home": ga5_h,
            "gf5_away": gf5_a, "ga_away_5": ga5_a,
            "xg5_home": xg5_h, "xga5_home": xga5_h,
            "xg5_away": xg5_a, "xga5_away": xga5_a,
            "form_diff": (ppg5_h - ppg5_a),
            "goals_exp_home": 0.5 * (xg5_h + xga5_a),
            "goals_exp_away": 0.5 * (xg5_a + xga5_h),
            "rest_home": min(rest_h, 30), "rest_away": min(rest_a, 30),
            "played_home": sh.played, "played_away": sa.played,
            "h2h_diff": h2h_diff,
            "league_goals": float(self.league_goals.get(div, 2.6)),
            "home_new": 1.0 if sh.played < 3 else 0.0,
            "away_new": 1.0 if sa.played < 3 else 0.0,
        }

    # ------------------------------------------------------------------ API
    def build(self, history: pd.DataFrame,
              fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
        """Retourne un DataFrame de features + labels pour l'historique,
        et les features des matchs a venir si `fixtures` est fourni.
        """
        rows: list[dict] = []
        if history is None or history.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        history = history.copy()
        history["_date"] = history["date"].map(_to_date)
        history = history.sort_values(["_date", "time"], kind="mergesort")

        for _, record in history.iterrows():
            home = canonical_team(record["home"])
            away = canonical_team(record["away"])
            div = str(record["div"])
            day = record["_date"]
            features = self._row_features(home, away, div, day)

            fthg = record.get("fthg")
            ftag = record.get("ftag")
            labels_ok = fthg is not None and ftag is not None and \
                not pd.isna(fthg) and not pd.isna(ftag)
            row = {
                "key": record.get("key"),
                "date": day.isoformat(),
                "div": div,
                "home": record["home"],
                "away": record["away"],
                **features,
            }
            if labels_ok:
                fthg, ftag = int(fthg), int(ftag)
                row.update({
                    "y_result": 0 if fthg > ftag else (1 if fthg == ftag else 2),
                    "y_home_goals": fthg, "y_away_goals": ftag,
                    "y_over25": int(fthg + ftag > 2.5),
                    "y_btts": int(fthg > 0 and ftag > 0),
                })
                # mise a jour de l'etat APRES avoir calcule les features
                self.teams[home].add(day, fthg, ftag,
                                     record.get("hxg"),
                                     record.get("axg"))
                self.teams[away].add(day, ftag, fthg,
                                     record.get("axg"),
                                     record.get("hxg"))
                self.h2h[(home, away)].append({"gf": fthg, "ga": ftag})
                self.h2h[(away, home)].append({"gf": ftag, "ga": fthg})
                self._update_elo(home, away, fthg, ftag)
                prev = self.league_goals.get(div, 2.6)
                self.league_goals[div] = prev + 0.02 * ((fthg + ftag) - prev)
            rows.append(row)

        if fixtures is not None and not fixtures.empty:
            fixtures = fixtures.copy()
            fixtures["_date"] = fixtures["date"].map(_to_date)
            for _, record in fixtures.sort_values("_date").iterrows():
                home = canonical_team(record["home"])
                away = canonical_team(record["away"])
                div = str(record["div"])
                day = record["_date"]
                row = {
                    "key": record.get("key"),
                    "date": day.isoformat(),
                    "div": div,
                    "home": record["home"],
                    "away": record["away"],
                    **self._row_features(home, away, div, day),
                }
                # Les matchs de test du backtest sont deja joues : on garde
                # leur label pour le reglement, SANS jamais mettre a jour
                # l'etat (pas de fuite dans les features).
                fthg = record.get("fthg")
                ftag = record.get("ftag")
                if fthg is not None and ftag is not None and \
                        not pd.isna(fthg) and not pd.isna(ftag):
                    fthg, ftag = int(fthg), int(ftag)
                    row.update({
                        "y_result": 0 if fthg > ftag else
                        (1 if fthg == ftag else 2),
                        "y_home_goals": fthg, "y_away_goals": ftag,
                        "y_over25": int(fthg + ftag > 2.5),
                        "y_btts": int(fthg > 0 and ftag > 0),
                    })
                else:
                    row.update({"y_result": np.nan, "y_home_goals": np.nan,
                                "y_away_goals": np.nan, "y_over25": np.nan,
                                "y_btts": np.nan})
                rows.append(row)
        return pd.DataFrame(rows)
