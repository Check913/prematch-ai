# -*- coding: utf-8 -*-
"""Detection de value : modele contre marche.

Principe (le seul qui tienne sur la duree) :
  1. on reconstruit la probabilite « juste » du marche en retirant la marge
     (consensus multi-bookmakers, Pinnacle en priorite) ;
  2. on compare a la probabilite du modele ;
  3. si p_modele * meilleure_cote - 1 > seuil, il y a de la valeur ;
  4. la mise suit un Kelly fractionne plafonne, jamais un pourcentage fixe.

Sans etape 1, un modele « gourmand » trouve de la value partout et perd.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterable

from .logging_setup import get_logger
from .markets import (ValuePick, devig, flat_stake, kelly_stake_fraction,
                      selection_label)

LOGGER = get_logger("prematch.value")

REFERENCE_PRIORITY = ["PS", "B365", "BFD", "BV", "PP", "SKB", "BW",
                      "DEMO", "ALT", "BFE", "AVG", "MAX"]
PSEUDO_BOOKS = {"AVG", "MAX"}

ALLOWED_MARKETS = {"1X2", "DC", "OU", "BTTS", "AH"}

EDGE_FLOOR_BY_MARKET = {"OU": 0.035, "BTTS": 0.035, "AH": 0.04}


# ---------------------------------------------------------------------------
# Lecture du marche
# ---------------------------------------------------------------------------


def market_probabilities(odds_rows: Iterable[Any],
                         method: str = "shin"
                         ) -> dict[tuple[str, str, float | None], float]:
    """Probabilites de-viggees du marche par (marche, selection, ligne)."""
    rows = [dict(row) if not isinstance(row, dict) else row for row in odds_rows]
    out: dict[tuple[str, str, float | None], float] = {}

    books: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("market") != "1X2":
            continue
        books.setdefault(str(row["book"]), {})[str(row["selection"])] = \
            float(row["price"])
    chosen = None
    for book in REFERENCE_PRIORITY:
        values = books.get(book, {})
        if {"H", "D", "A"} <= set(values):
            chosen = values
            break
    if chosen is None:
        for values in books.values():
            if {"H", "D", "A"} <= set(values):
                chosen = values
                break
    if chosen:
        try:
            probs = devig([chosen["H"], chosen["D"], chosen["A"]], method)
            out[("1X2", "H", None)] = probs[0]
            out[("1X2", "D", None)] = probs[1]
            out[("1X2", "A", None)] = probs[2]
            out[("DC", "1X", None)] = probs[0] + probs[1]
            out[("DC", "12", None)] = probs[0] + probs[2]
            out[("DC", "X2", None)] = probs[1] + probs[2]
        except ValueError:
            pass

    for market, two_way in (("OU", ("OVER", "UNDER")), ("AH", ("HOME", "AWAY"))):
        per_line: dict[float, dict[str, dict[str, float]]] = {}
        for row in rows:
            if row.get("market") != market or row.get("line") is None:
                continue
            line = float(row["line"])
            per_line.setdefault(line, {}).setdefault(
                str(row["book"]), {})[str(row["selection"])] = float(row["price"])
        for line, per_book in per_line.items():
            picked = None
            for book in REFERENCE_PRIORITY:
                values = per_book.get(book, {})
                if set(two_way) <= set(values):
                    picked = values
                    break
            if picked is None:
                continue
            try:
                probs = devig([picked[two_way[0]], picked[two_way[1]]], "power")
            except ValueError:
                continue
            out[(market, two_way[0], line)] = probs[0]
            out[(market, two_way[1], line)] = probs[1]
    return out


def best_prices(odds_rows: Iterable[Any]
                ) -> dict[tuple[str, str, float | None], tuple[float, str]]:
    """Meilleure cote disponible par selection (pseudo-books en repli)."""
    best: dict[tuple[str, str, float | None], tuple[float, str]] = {}
    fallback: dict[tuple[str, str, float | None], tuple[float, str]] = {}
    for raw in odds_rows:
        row = dict(raw) if not isinstance(raw, dict) else raw
        if row.get("market") not in ALLOWED_MARKETS:
            continue
        line = float(row["line"]) if row.get("line") is not None else None
        key = (str(row["market"]), str(row["selection"]), line)
        price = float(row["price"])
        if price <= 1.01:
            continue
        book = str(row.get("book") or "?")
        target = fallback if book in PSEUDO_BOOKS else best
        if key not in target or price > target[key][0]:
            target[key] = (price, book)
    for key, value in fallback.items():
        best.setdefault(key, value)
    return best


def agreement(p_dc: Iterable[float], p_gbm: Iterable[float]) -> float:
    """1 - distance de variation totale entre les deux modeles (0..1)."""
    left, right = list(p_dc), list(p_gbm)
    if not left or not right or len(left) != len(right):
        return 1.0
    return float(max(0.0, 1.0 - 0.5 * sum(abs(a - b)
                                          for a, b in zip(left, right))))


# ---------------------------------------------------------------------------
# Analyse d'un match : toutes les selections chiffrees
# ---------------------------------------------------------------------------


def market_edges(model_row: dict, odds_rows: list[Any],
                 devig_method: str = "shin",
                 edge_cap: float | None = None) -> list[dict]:
    """Liste triee des selections : cote, proba modele, proba marche, edge.

    Si `edge_cap` est fourni, les selections au-dela sont marquees
    `suspicious=True` (visibles dans l'analyse mais non retenues).
    """
    if not odds_rows:
        return []
    prices = best_prices(odds_rows)
    market_probs = market_probabilities(odds_rows, devig_method)
    edges: list[dict] = []
    for key, (price, book) in prices.items():
        market, selection, line = key
        p_model = model_row.get("markets", {}).get(key)
        p_market = market_probs.get(key)
        if p_model is None or p_market is None or p_market <= 0:
            continue
        edge = float(p_model) * float(price) - 1.0
        edges.append({
            "market": market, "selection": selection, "line": line,
            "price": float(price), "book": book,
            "p_model": float(p_model), "p_market": float(p_market),
            "edge": edge,
            "suspicious": edge_cap is not None and edge > edge_cap,
            "fair_price": round(1.0 / max(p_model, 1e-6), 2),
        })
    edges.sort(key=lambda item: -item["edge"])
    return edges


def analyse_match(model_row: dict, odds_rows: list[Any],
                  devig_method: str = "shin") -> dict:
    """Vue complete d'un match (pour le digest et les alternatives)."""
    conf = agreement(model_row.get("p_dc", []), model_row.get("p_gbm", []))
    return {
        "key": model_row.get("key"),
        "home": model_row.get("home"),
        "away": model_row.get("away"),
        "div": model_row.get("div"),
        "kickoff": model_row.get("kickoff", ""),
        "p_1x2": [float(x) for x in model_row.get("p_1x2", [])],
        "p_dc": [float(x) for x in model_row.get("p_dc", [])],
        "p_gbm": [float(x) for x in model_row.get("p_gbm", [])],
        "agreement": conf,
        "edges": market_edges(model_row, odds_rows, devig_method),
    }


# ---------------------------------------------------------------------------
# Selection des paris
# ---------------------------------------------------------------------------


def build_picks(model_rows: list[dict], odds_by_match: dict[str, list[Any]],
                cfg: Any, bankroll: float,
                devig_method: str = "shin") -> list[ValuePick]:
    """Transforme les probabilites du modele en paris candidats filtres."""
    app = cfg.section("app")
    min_edge = float(app.get("min_edge", 0.03))
    min_price = float(app.get("min_price", 1.40))
    max_price = float(app.get("max_price", 8.0))
    min_prob = float(app.get("min_model_prob", 0.15))
    min_agree = float(app.get("min_agreement", 0.55))
    max_per_match = int(app.get("max_picks_per_match", 1))
    kelly_fraction = float(app.get("kelly_fraction", 0.25))
    max_stake_pct = float(app.get("max_stake_pct", 2.0))
    # Un edge enorme (>60%) est presque toujours un artefact du modele
    # (equipe peu de donnees, donnees douteuses), pas de l'argent libre :
    # on ecarte ces signaux au lieu de parier dessus.
    edge_cap = float(app.get("max_edge", 0.60))
    allowed = set(app.get("markets", ["1X2", "DC", "OU", "BTTS", "AH"]))

    candidates: list[ValuePick] = []
    for row in model_rows:
        odds_rows = odds_by_match.get(row["key"], [])
        if not odds_rows:
            continue
        conf = agreement(row.get("p_dc", []), row.get("p_gbm", []))
        if conf < min_agree:
            continue
        for item in market_edges(row, odds_rows, devig_method):
            if item["market"] not in allowed:
                continue
            price = item["price"]
            if not (min_price <= price <= max_price):
                continue
            if item["p_model"] < min_prob:
                continue
            floor = max(min_edge, EDGE_FLOOR_BY_MARKET.get(item["market"], 0.0))
            if item["edge"] < floor:
                continue
            if item["edge"] > edge_cap:
                LOGGER.info("edge suspect ignore (%.1f%%) : %s %s",
                            item["edge"] * 100, row.get("key"), item)
                continue
            stake = flat_stake(item["p_model"], price, bankroll, kelly_fraction,
                               max_stake_pct)
            if stake <= 0:
                continue
            candidates.append(ValuePick(
                match_key=row["key"], home=row.get("home", ""),
                away=row.get("away", ""), div=row.get("div", ""),
                kickoff=row.get("kickoff", ""), market=item["market"],
                selection=item["selection"], line=item["line"], price=price,
                book=item["book"], p_model=item["p_model"],
                p_market=item["p_market"], edge=item["edge"],
                kelly=kelly_stake_fraction(item["p_model"], price,
                                           kelly_fraction),
                stake=stake, confidence=conf))

    candidates.sort(key=lambda p: (-p.edge, -p.confidence))
    selected: list[ValuePick] = []
    per_match: dict[str, int] = {}
    for pick in candidates:
        if per_match.get(pick.match_key, 0) >= max_per_match:
            continue
        per_match[pick.match_key] = per_match.get(pick.match_key, 0) + 1
        selected.append(pick)

    max_exposure = bankroll * float(
        app.get("daily_max_exposure_pct", 10.0)) / 100.0
    total = 0.0
    capped: list[ValuePick] = []
    for pick in selected:
        if total + pick.stake > max_exposure:
            remaining = round(max_exposure - total, 2)
            if remaining < 1.0:
                continue
            pick = dataclasses.replace(pick, stake=remaining)
        total += pick.stake
        capped.append(pick)

    max_picks = int(app.get("max_picks_per_day", 8))
    return capped[:max_picks]


def picks_to_rows(picks: Iterable[ValuePick], model_version: str) -> list[dict]:
    """Convertit les picks en lignes pour la table `predictions`."""
    return [{
        "match_key": p.match_key, "kickoff": p.kickoff, "div": p.div,
        "home": p.home, "away": p.away, "model_version": model_version,
        "market": p.market, "selection": p.selection, "line": p.line,
        "p_model": p.p_model, "p_market": p.p_market, "price": p.price,
        "book": p.book, "edge": p.edge, "kelly": p.kelly, "stake": p.stake,
        "confidence": p.confidence,
    } for p in picks]


def label_for(pick: ValuePick) -> str:
    return selection_label(pick.market, pick.selection, pick.line,
                           pick.home, pick.away)
