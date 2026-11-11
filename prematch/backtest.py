# -*- coding: utf-8 -*-
"""Backtest walk-forward : le seul chiffre qui compte.

On re-joue l'historique exactement comme le systeme vivrait chaque jour :
  * au jour D, le modele ne connait QUE les matchs avant D ;
  * il entraine, puis mise (virtuellement) sur les matchs du jour D
    dont l'edge depasse le seuil, avec les cotes de fermeture
    (worst case honnete : c'est ce que le marche pensait au coup d'envoi) ;
  * on mesure ROI, hit-rate, profit par pari, courbe de bankroll,
    ecart-type journee -> Sharpe approximatif.

Si le ROI cumulatif et le profit journe moyen restent positifs sur une
longue periode (>= 1 saison), la strategie a une vraie esperance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .db import settle_bet
from .features import FeatureBuilder, canonical_team
from .logging_setup import get_logger
from .markets import (devig, flat_stake, kelly_stake_fraction,
                      matrix_markets, poisson_matrix)
from .model import ModelBundle, _blend
from .value import market_probabilities

LOGGER = get_logger("prematch.backtest")


@dataclass
class BacktestResult:
    n_days: int
    n_matches: int
    n_bets: int
    n_won: int
    n_lost: int
    n_void: int
    stake_total: float
    pnl_total: float
    roi_pct: float
    hit_rate: float
    avg_profit_per_bet: float
    sharpe_daily: float
    max_drawdown_pct: float
    final_bankroll: float
    bet_log: pd.DataFrame
    equity: pd.Series

    def summary(self) -> str:
        return (f"{self.n_bets} paris sur {self.n_matches} matchs "
                f"({self.n_days} jours) | ROI {self.roi_pct:+.2f}% | "
                f"reussite {self.hit_rate * 100:.1f}% | "
                f"P&L {self.pnl_total:+.2f} | "
                f"drawdown max {self.max_drawdown_pct:.1f}% | "
                f"Sharpe j. {self.sharpe_daily:.2f}")


def _edges_for_match(model_row: dict, odds_rows: list[dict],
                     min_edge: float, devig_method: str,
                     edge_cap: float = 0.60) -> list[dict]:
    """Selections avec edge positif (cotes de fermeture uniquement)."""
    if not odds_rows:
        return []
    probs_market = market_probabilities(odds_rows, devig_method)
    best: dict[tuple[str, str, float | None], tuple[float, str]] = {}
    for row in odds_rows:
        key = (row["market"], row["selection"],
               float(row["line"]) if row.get("line") is not None else None)
        price = float(row["price"])
        if price <= 1.01:
            continue
        if key not in best or price > best[key][0]:
            best[key] = (price, str(row["book"]))
    edges = []
    for key, (price, book) in best.items():
        p_model = model_row.get("markets", {}).get(key)
        p_market = probs_market.get(key)
        if p_model is None or p_market is None or p_market <= 0:
            continue
        edge = p_model * price - 1.0
        if edge >= min_edge and edge <= edge_cap:
            edges.append({"market": key[0], "selection": key[1],
                          "line": key[2], "price": price, "book": book,
                          "p_model": p_model, "p_market": p_market,
                          "edge": edge})
    edges.sort(key=lambda item: -item["edge"])
    return edges


def run_walk_forward(db, cfg: Config, seasons: list[str] | None = None,
                     train_days: int = 365, test_days: int = 45,
                     min_edge: float | None = None,
                     n_matches_max: int = 8) -> BacktestResult:
    """Backtest glissant : re-entrainement par blocs de `test_days` jours.

    `train_days` : fenetre d'historique utilisee a chaque bloc (decalage
    temporel respecte : aucun match de la fenetre de test ne fuite dans
    l'entrainement).
    """
    app = cfg.section("app")
    min_edge = float(min_edge if min_edge is not None
                     else app.get("min_edge", 0.03))
    bankroll0 = float(app.get("bankroll_eur", 1000.0))
    kelly_fraction = float(app.get("kelly_fraction", 0.25))
    max_stake_pct = float(app.get("max_stake_pct", 2.0))
    devig_method = str(cfg.get("model.devig_method", "shin"))
    min_price = float(app.get("min_price", 1.40))
    max_price = float(app.get("max_price", 8.0))
    edge_cap = float(app.get("max_edge", 0.60))

    rows = [dict(r) for r in db.finished_matches()]
    if not rows:
        raise RuntimeError("pas d'historique en base")
    frame = pd.DataFrame(rows)
    frame["_d"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("_d")

    all_odds = pd.DataFrame([dict(r) for r in db.conn.execute(
        "SELECT o.*, m.status FROM odds o JOIN matches m ON m.key=o.match_key "
        "WHERE o.is_closing=1")])
    if all_odds.empty:
        raise RuntimeError("pas de cotes de fermeture en base")

    end = frame["_d"].max()
    start = end - timedelta(days=test_days)
    while start > frame["_d"].min() + timedelta(days=train_days):
        start -= timedelta(days=test_days)
    days = pd.date_range(start, end, freq=f"{test_days}D")
    if days[-1] < end:
        days = days.append(pd.DatetimeIndex([end]))

    stake_total = pnl_total = 0.0
    n_bets = n_won = n_lost = n_void = 0
    daily_pnl: dict[str, float] = {}
    bet_log: list[dict] = []
    bankroll = bankroll0
    equity_points: list[float] = [bankroll]

    for block_start, block_end in zip(days[:-1], days[1:]):
        block_end = block_end.to_pydatetime().date()
        block_start = block_start.to_pydatetime().date()
        train = frame[(frame["_d"].dt.date < block_start)
                      & (frame["_d"].dt.date >= block_start
                         - timedelta(days=train_days))]
        test = frame[(frame["_d"].dt.date >= block_start)
                     & (frame["_d"].dt.date < block_end)]
        if train.empty or test.empty:
            equity_points.append(bankroll)
            continue

        builder = FeatureBuilder()
        features = builder.build(train)
        # holdout interne = derniers 45 j de la fenetre d'entrainement :
        # ajuste le poids ET la temperature sans aucune fuite vers le bloc
        # de test (le holdout est strictement anterieur au bloc).
        bundle = ModelBundle.train(train, features, cfg, holdout_days=45)

        # une seule passe : l'historique alimente l'etat, les matchs de test
        # sont transformes SANS mettre a jour cetetat (pas de fuite).
        upcoming = builder.build(train, test)
        labelled = upcoming.dropna(subset=["y_result"])
        if labelled.empty:
            equity_points.append(bankroll)
            continue

        results = bundle.market_probabilities(labelled)
        test_keys = set(labelled["key"])
        closing = all_odds[all_odds["match_key"].isin(test_keys)]
        odds_by_match = {key: group.to_dict("records")
                         for key, group in closing.groupby("match_key")}

        for item in results:
            edges = _edges_for_match(item, odds_by_match.get(item["key"], []),
                                     min_edge, devig_method,
                                     edge_cap)[:n_matches_max]
            for edge in edges:
                price = edge["price"]
                if not (min_price <= price <= max_price):
                    continue
                stake = flat_stake(edge["p_model"], price, bankroll,
                                   kelly_fraction, max_stake_pct)
                if stake < 1.0:
                    continue
                match_row = test[test["key"] == item["key"]]
                if match_row.empty:
                    continue
                match_row = match_row.iloc[0]
                outcome = settle_bet(edge["market"], edge["selection"],
                                     edge["line"], int(match_row["fthg"]),
                                     int(match_row["ftag"]))
                if outcome == "won":
                    pnl = stake * (price - 1.0)
                    bankroll += pnl
                    n_won += 1
                elif outcome == "lost":
                    pnl = -stake
                    bankroll -= stake
                    n_lost += 1
                else:
                    pnl = 0.0
                    n_void += 1
                n_bets += 1
                stake_total += stake
                pnl_total += pnl
                day = str(match_row["date"])
                daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
                equity_points.append(bankroll)
                bet_log.append({
                    "date": day, "key": item["key"],
                    "match": f"{item['home']} v {item['away']}",
                    "market": edge["market"], "selection": edge["selection"],
                    "line": edge["line"], "price": price, "book": edge["book"],
                    "p_model": round(edge["p_model"], 4),
                    "p_market": round(edge["p_market"], 4),
                    "edge": round(edge["edge"], 4), "stake": stake,
                    "outcome": outcome, "pnl": round(pnl, 2),
                    "bankroll": round(bankroll, 2)})

    equity = pd.Series(equity_points, dtype=float)
    if len(equity) > 2:
        peak = equity.cummax()
        max_drawdown_pct = float(((equity - peak) / peak).min() * 100.0)
    else:
        max_drawdown_pct = 0.0
    daily = pd.Series(list(daily_pnl.values()), dtype=float)
    sharpe = float(daily.mean() / daily.std() * (252 ** 0.5)) \
        if len(daily) > 2 and daily.std() > 0 else 0.0

    return BacktestResult(
        n_days=len(daily_pnl), n_matches=len(frame[frame["_d"].dt.date >=
                                                   days[0].to_pydatetime().date()]),
        n_bets=n_bets, n_won=n_won, n_lost=n_lost, n_void=n_void,
        stake_total=round(stake_total, 2), pnl_total=round(pnl_total, 2),
        roi_pct=round(pnl_total / stake_total * 100, 2) if stake_total else 0.0,
        hit_rate=round(n_won / max(1, n_won + n_lost), 4),
        avg_profit_per_bet=round(pnl_total / max(1, n_bets), 2),
        sharpe_daily=round(sharpe, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        final_bankroll=round(bankroll, 2),
        bet_log=pd.DataFrame(bet_log), equity=equity)


def edge_sweep(db, cfg: Config, edges: list[float] | None = None
               ) -> list[dict[str, Any]]:
    """Teste plusieurs seuils d'edge et retourne le tableau de resultats."""
    edges = edges or list(cfg.section("learning").get(
        "tune_min_edge_grid", [0.02, 0.03, 0.04, 0.05]))
    results = []
    for threshold in edges:
        try:
            outcome = run_walk_forward(db, cfg, min_edge=float(threshold))
            results.append({"min_edge": threshold, "summary": outcome.summary(),
                            "roi_pct": outcome.roi_pct,
                            "n_bets": outcome.n_bets,
                            "sharpe": outcome.sharpe_daily})
            LOGGER.info("edge>=%.2f : %s", threshold, outcome.summary())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("backtest edge=%.2f echoue : %s", threshold, exc)
            results.append({"min_edge": threshold, "error": str(exc)})
    return results
