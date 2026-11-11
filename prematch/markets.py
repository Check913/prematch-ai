# -*- coding: utf-8 -*-
"""Mathematiques des marches : devig, matrice de scores, EV, Kelly.

Trois blocs :
  1. passage cote -> probabilite (multiplicative, additive, power, Shin) ;
  2. matrice de scores Dixon-Coles -> probabilites de tous les marches
     (1N2, double chance, plus/moins, BTTS, handicap, score exact) ;
  3. valorisation : valeur attendue, Kelly fractionne, cote equitable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# 1. Cotes -> probabilites
# ---------------------------------------------------------------------------


def overround(odds: Sequence[float]) -> float:
    """Marge du bookmaker : somme des probabilites implicites - 1."""
    return sum(1.0 / o for o in odds if o and o > 1.0) - 1.0


def devig(odds: Sequence[float], method: str = "shin") -> list[float]:
    """Retire la marge (vig) d'un ensemble de cotes exhaustif.

    method :
      * 'multiplicative' : normalisation simple (rapide, sous-estime les
        favoris) ;
      * 'additive' : retrait uniforme de la marge ;
      * 'power' : p_i = q_i^k avec k resolu pour sommer a 1 ;
      * 'shin' : modele de Shin (proportion d'informes), le plus fiable
        sur les marches 1N2 avec favoris marques.
    """
    prices = [float(o) for o in odds]
    if any(o <= 1.0 for o in prices):
        raise ValueError(f"cotes invalides : {prices}")
    q = np.array([1.0 / o for o in prices], dtype=float)
    total = float(q.sum())
    n = len(prices)
    method = (method or "").lower()

    if method == "multiplicative":
        return [float(x / total) for x in q]
    if method == "additive":
        return [float(max(1e-6, x - (total - 1.0) / n)) for x in q]
    if method == "power":
        if abs(total - 1.0) < 1e-12:
            return [float(x) for x in q]

        def f(k: float) -> float:
            return float(np.sum(q ** k)) - 1.0

        try:
            k = brentq(f, 0.5, 3.0, xtol=1e-10)
        except ValueError:
            return [float(x / total) for x in q]
        return [float(x ** k) for x in q]
    if method == "shin":
        if abs(total - 1.0) < 1e-12:
            return [float(x) for x in q]

        def probs(z: float) -> np.ndarray:
            return (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / total) - z) \
                / (2.0 * (1.0 - z))

        def g(z: float) -> float:
            return float(probs(z).sum()) - 1.0

        try:
            z = brentq(g, 0.0, 0.45, xtol=1e-10)
        except ValueError:
            return [float(x / total) for x in q]
        p = probs(z)
        return [float(x) for x in p / p.sum()]
    raise ValueError(f"methode de devig inconnue : {method}")


def implied_probability(price: float) -> float:
    return 1.0 / float(price) if price and float(price) > 1.0 else float("nan")


def fair_price(prob: float, margin: float = 0.0) -> float:
    """Cote equitable (+ marge eventuelle)."""
    if not prob or prob <= 0:
        return float("inf")
    return round(1.0 / (prob * (1.0 + margin)), 3)


# ---------------------------------------------------------------------------
# 2. Matrice de scores -> marches
# ---------------------------------------------------------------------------


def poisson_matrix(home_rate: float, away_rate: float,
                   rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """Matrice P(i,j) Dixon-Coles des scores (i buts domicile, j exterieur)."""
    home_rate = max(0.02, float(home_rate))
    away_rate = max(0.02, float(away_rate))
    home = poisson.pmf(np.arange(max_goals + 1), home_rate)
    away = poisson.pmf(np.arange(max_goals + 1), away_rate)
    matrix = np.outer(home, away)
    if rho:
        matrix = matrix.astype(float)
        lam, mu = home_rate, away_rate
        matrix[0, 0] *= 1.0 - lam * mu * rho
        matrix[0, 1] *= 1.0 + lam * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho
        matrix = np.clip(matrix, 1e-12, None)
    return matrix / matrix.sum()


def matrix_markets(matrix: np.ndarray, ou_lines: Iterable[float] =
                   (1.5, 2.5, 3.5), ah_lines: Iterable[float] =
                   (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
                   top_scores: int = 6) -> dict[tuple[str, str, float | None],
                                                float]:
    """Derive les probabilites de tous les marches depuis la matrice.

    La cle est (marche, selection, ligne). Pour les handicaps, la ligne est
    du point de vue de l'equipe a domicile.
    """
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    goals_home = np.arange(size)[:, None]
    goals_away = np.arange(size)[None, :]
    margin = goals_home - goals_away  # >0 : victoire domicile

    out: dict[tuple[str, str, float | None], float] = {}
    out[("1X2", "H", None)] = float(matrix[margin > 0].sum())
    out[("1X2", "D", None)] = float(matrix[margin == 0].sum())
    out[("1X2", "A", None)] = float(matrix[margin < 0].sum())

    p_home, p_draw, p_away = (out[("1X2", "H", None)],
                              out[("1X2", "D", None)],
                              out[("1X2", "A", None)])
    out[("DC", "1X", None)] = p_home + p_draw
    out[("DC", "12", None)] = p_home + p_away
    out[("DC", "X2", None)] = p_draw + p_away

    total_goals = goals_home + goals_away
    for line in ou_lines:
        over = float(matrix[total_goals > float(line)].sum())
        out[("OU", "OVER", float(line))] = over
        out[("OU", "UNDER", float(line))] = 1.0 - over

    btts = float(matrix[(goals_home > 0) & (goals_away > 0)].sum())
    out[("BTTS", "YES", None)] = btts
    out[("BTTS", "NO", None)] = 1.0 - btts

    for line in ah_lines:
        # value = marge + ligne ; >0 gagne, =0 rembourse
        value = margin + float(line)
        out[("AH", "HOME", float(line))] = float(matrix[value > 1e-9].sum())
        out[("AH", "AWAY", float(line))] = float(matrix[value < -1e-9].sum())
        out[("AH", "PUSH", float(line))] = float(
            matrix[np.abs(value) <= 1e-9].sum())

    flat = [(int(i), int(j), float(matrix[i, j]))
            for i in range(size) for j in range(size)]
    flat.sort(key=lambda x: -x[2])
    for home_goals, away_goals, prob in flat[:top_scores]:
        out[("CS", f"{home_goals}-{away_goals}", None)] = prob

    return out


def rates_from_probs(p_home: float, p_draw: float, p_away: float,
                     p_over25: float, total_line: float = 2.5,
                     max_goals: int = 10) -> tuple[float, float]:
    """Retrouve (lambda_domicile, lambda_exterieur) coherents avec les
    probabilites finales 1N2 et Plus/Moins 2.5.

    Deux dichotomies suffisent (rapide) :
      1. total de buts : inversion de P(total > 2.5) sur une Poisson ;
      2. ecart de buts (supremacy) : on cale p_domicile - p_exterieur.
    """
    p_home = float(np.clip(p_home, 1e-4, 1 - 1e-4))
    p_away = float(np.clip(p_away, 1e-4, 1 - 1e-4))
    target_over = float(np.clip(p_over25, 1e-3, 1 - 1e-3))

    lo, hi = 0.2, 7.5
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if (1.0 - poisson.cdf(int(np.floor(total_line)), mid)) < target_over:
            lo = mid
        else:
            hi = mid
    total = float(np.clip(0.5 * (lo + hi), 0.8, 5.5))

    target_diff = p_home - p_away

    def home_minus_away(lam_h: float, lam_a: float) -> float:
        matrix = poisson_matrix(lam_h, lam_a, 0.0, max_goals)
        size = matrix.shape[0]
        gh = np.arange(size)[:, None]
        ga = np.arange(size)[None, :]
        m = gh - ga
        return float(matrix[m > 0].sum() - matrix[m < 0].sum())

    lo_s, hi_s = -total + 0.15, total - 0.15
    for _ in range(40):
        mid = 0.5 * (lo_s + hi_s)
        lam_h = max(0.05, (total + mid) / 2.0)
        lam_a = max(0.05, (total - mid) / 2.0)
        if home_minus_away(lam_h, lam_a) < target_diff:
            lo_s = mid
        else:
            hi_s = mid
    supremacy = 0.5 * (lo_s + hi_s)
    lam_h = max(0.05, (total + supremacy) / 2.0)
    lam_a = max(0.05, (total - supremacy) / 2.0)
    return lam_h, lam_a


# ---------------------------------------------------------------------------
# 3. Valorisation
# ---------------------------------------------------------------------------


def expected_value(prob: float, price: float) -> float:
    """EV par unite misee : p*(cote-1) - (1-p)."""
    return float(prob) * (float(price) - 1.0) - (1.0 - float(prob))


def edge(prob: float, price: float) -> float:
    """Rendement attendu relatif : p*cote - 1."""
    return float(prob) * float(price) - 1.0


def kelly_stake_fraction(prob: float, price: float,
                         fraction: float = 0.25) -> float:
    """Fraction de bankroll Kelly (0 si pas de valeur)."""
    p = float(prob)
    o = float(price)
    if o <= 1.0 or p <= 0.0 or p >= 1.0:
        return 0.0
    b = o - 1.0
    f = (p * b - (1.0 - p)) / b
    return max(0.0, f * float(fraction))


def flat_stake(prob: float, price: float, bankroll: float,
               kelly_fraction: float = 0.25, max_stake_pct: float = 2.0,
               min_stake: float = 1.0) -> float:
    """Mise finale : Kelly fractionne, plafonnee en % de bankroll."""
    raw = kelly_stake_fraction(prob, price, kelly_fraction) * bankroll
    capped = min(raw, bankroll * max_stake_pct / 100.0)
    if capped < min_stake:
        return 0.0
    return round(capped, 2)


# ---------------------------------------------------------------------------
# 4. Libelles (affichage Telegram)
# ---------------------------------------------------------------------------

MARKET_NAMES_FR = {
    "1X2": "Resultat du match",
    "DC": "Double chance",
    "OU": "Total de buts",
    "BTTS": "Les deux equipes marquent",
    "AH": "Handicap asiatique",
    "CS": "Score exact",
}

SELECTION_NAMES_FR = {
    ("1X2", "H"): "Victoire {home}",
    ("1X2", "D"): "Match nul",
    ("1X2", "A"): "Victoire {away}",
    ("DC", "1X"): "{home} ou nul",
    ("DC", "12"): "{home} ou {away}",
    ("DC", "X2"): "Nul ou {away}",
    ("BTTS", "YES"): "Oui",
    ("BTTS", "NO"): "Non",
    ("AH", "HOME"): "Handicap {line:+g} {home}",
    ("AH", "AWAY"): "Handicap {line:+g} {away}",
    ("OU", "OVER"): "Plus de {line:g} buts",
    ("OU", "UNDER"): "Moins de {line:g} buts",
    ("CS", "H"): "Score exact",
}


def selection_label(market: str, selection: str, line: float | None,
                    home: str = "domicile", away: str = "exterieur") -> str:
    """Libelle francais lisible d'une selection."""
    market = (market or "").upper()
    selection = (selection or "").upper()
    if market == "CS":
        return f"Score exact {selection}"
    template = SELECTION_NAMES_FR.get((market, selection))
    if template:
        return template.format(home=home, away=away,
                               line=float(line) if line is not None else 0.0)
    base = MARKET_NAMES_FR.get(market, market)
    suffix = f" {line:g}" if line is not None else ""
    return f"{base} - {selection}{suffix}"


@dataclass(frozen=True)
class ValuePick:
    """Un pari candidat avec toutes ses metriques."""

    match_key: str
    home: str
    away: str
    div: str
    kickoff: str
    market: str
    selection: str
    line: float | None
    price: float
    book: str
    p_model: float
    p_market: float
    edge: float
    kelly: float
    stake: float
    confidence: float

    @property
    def label(self) -> str:
        return selection_label(self.market, self.selection, self.line,
                              self.home, self.away)
