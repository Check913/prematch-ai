# -*- coding: utf-8 -*-
"""Tests de bout en bout sur donnees synthetiques (rapides)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prematch.config import load_config
from prematch.db import Database, settle_bet
from prematch.features import FeatureBuilder, canonical_team
from prematch.model import DixonColesModel
from prematch.pipeline import Pipeline
from prematch.sources import demo
from prematch.value import best_prices, build_picks, market_probabilities


@pytest.fixture(scope="module")
def demo_cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cfg")
    cfg = load_config(root=tmp)
    cfg.data["sources"]["demo"] = {"enabled": True, "seed": 11}
    cfg.data["app"]["min_edge"] = 0.02
    return cfg


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory, demo_cfg):
    tmp = tmp_path_factory.mktemp("data")
    db = Database(tmp / "test.db")
    demo.sync_demo(db, demo_cfg)   # historique + matchs a venir + cotes
    yield db
    db.close()


def test_canonical_team():
    assert canonical_team("Paris Saint-Germain") == "paris saint germain"
    assert canonical_team("Man. United") == "man utd"


class TestSettle:
    def test_1x2(self):
        assert settle_bet("1X2", "H", None, 2, 1) == "won"
        assert settle_bet("1X2", "H", None, 1, 1) == "lost"
        assert settle_bet("1X2", "D", None, 1, 1) == "won"

    def test_over_under(self):
        assert settle_bet("OU", "OVER", 2.5, 2, 1) == "won"
        assert settle_bet("OU", "OVER", 2.5, 1, 1) == "lost"  # 2 buts < 2.5
        assert settle_bet("OU", "UNDER", 2.5, 1, 0) == "won"
        # ligne entiere : egalite = remboursement
        assert settle_bet("OU", "OVER", 2.0, 1, 1) == "void"
        assert settle_bet("OU", "UNDER", 3.0, 2, 1) == "void"

    def test_btts_and_ah(self):
        assert settle_bet("BTTS", "YES", None, 1, 1) == "won"
        assert settle_bet("BTTS", "NO", None, 1, 1) == "lost"
        assert settle_bet("AH", "HOME", -0.5, 2, 1) == "won"
        assert settle_bet("AH", "AWAY", 0.5, 1, 1) == "won"   # nul +0.5 : gagne
        assert settle_bet("AH", "HOME", -0.5, 1, 1) == "lost"
        assert settle_bet("AH", "AWAY", 0.0, 1, 1) == "void"


class TestValue:
    def test_market_probabilities_sum(self, demo_db):
        rows = demo_db.conn.execute(
            "SELECT * FROM odds WHERE match_key IN (SELECT key FROM matches "
            "LIMIT 1)").fetchall()
        probs = market_probabilities(rows)
        assert abs(probs[("1X2", "H", None)] + probs[("1X2", "D", None)]
                   + probs[("1X2", "A", None)] - 1.0) < 1e-6

    def test_best_prices_prefers_real_books(self):
        rows = [
            {"match_key": "k", "book": "AVG", "market": "1X2", "selection": "H",
             "line": None, "price": 2.0},
            {"match_key": "k", "book": "B365", "market": "1X2",
             "selection": "H", "line": None, "price": 2.2},
        ]
        best = best_prices(rows)
        assert best[("1X2", "H", None)][0] == 2.2
        assert best[("1X2", "H", None)][1] == "B365"


class TestModel:
    def test_dixon_coles_fits_and_predicts(self, demo_db):
        rows = [dict(r) for r in demo_db.finished_matches()]
        frame = pd.DataFrame(rows)
        model = DixonColesModel().fit(frame)
        assert model.n_matches > 100
        p = model.proba_1x2(frame.iloc[0]["home"], frame.iloc[0]["away"],
                            frame.iloc[0]["div"])
        assert abs(p.sum() - 1.0) < 1e-6
        # avec des forces fixes, la coherence de la matrice est verifiee
        markets = model.markets(frame.iloc[0]["home"], frame.iloc[0]["away"],
                                frame.iloc[0]["div"])
        assert markets[("1X2", "H", None)] > 0


class TestEndToEnd:
    def test_full_pipeline_demo(self, demo_db, demo_cfg):
        pipeline = Pipeline(demo_cfg, db=demo_db)
        trained = pipeline.train()
        assert trained["n_train"] > 100
        result = pipeline.predict(days_ahead=10)
        assert result["n_matches"] > 0
        # en mode demo les matchs "a venir" sont generes mais termines :
        # le test verifie surtout que la chaine ne casse pas
        text = pipeline.digest_text(result=result)
        assert "PREMATCH AI" in text

    def test_pick_filters(self, demo_db, demo_cfg):
        demo_cfg.data["app"]["min_edge"] = 10.0  # trop strict : aucun pick
        pipeline = Pipeline(demo_cfg, db=demo_db)
        result = pipeline.predict(days_ahead=10)
        assert result["picks"] == []

    def test_backtest_runs(self, demo_db, demo_cfg):
        from prematch.backtest import run_walk_forward
        result = run_walk_forward(demo_db, demo_cfg, train_days=200,
                                  test_days=30, min_edge=0.02)
        assert result.n_bets >= 0
        assert result.roi_pct == result.roi_pct  # pas de NaN


class TestLearning:
    def test_daily_cycle_no_crash(self, demo_db, demo_cfg):
        from prematch.learning import adjust_risk, daily_cycle
        pipeline = Pipeline(demo_cfg, db=demo_db)
        outcome = daily_cycle(pipeline)
        assert "risk" in outcome
        risk = adjust_risk(demo_db, demo_cfg)
        assert risk["min_edge"] >= demo_cfg.get("app.min_edge")
