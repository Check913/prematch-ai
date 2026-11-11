# -*- coding: utf-8 -*-
"""Tests des mathematiques de marches (devig, matrice, Kelly)."""
from __future__ import annotations

import numpy as np
import pytest

from prematch.markets import (devig, edge, expected_value, flat_stake,
                              kelly_stake_fraction, matrix_markets,
                              overround, poisson_matrix, rates_from_probs,
                              selection_label)


class TestDevig:
    def test_multiplicative_sums_to_one(self):
        probs = devig([2.0, 3.5, 3.8], "multiplicative")
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_shin_sums_to_one(self):
        probs = devig([1.5, 4.0, 6.5], "shin")
        assert abs(sum(probs) - 1.0) < 1e-9
        # le favori doit avoir une probabilite plus elevee qu'en normalisation
        assert probs[0] > 1 / 1.5 / (1 / 1.5 + 1 / 4.0 + 1 / 6.5)

    def test_power_fair_market(self):
        probs = devig([2.0, 2.0], "power")
        assert abs(probs[0] - 0.5) < 1e-6

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError):
            devig([1.0, 3.0, 4.0])

    def test_overround(self):
        assert overround([2.0, 2.0]) == pytest.approx(0.0)
        assert overround([2.0, 3.5, 3.8]) > 0


class TestMatrix:
    def test_matrix_sums_to_one(self):
        matrix = poisson_matrix(1.5, 1.2, -0.05, 10)
        assert abs(matrix.sum() - 1.0) < 1e-6

    def test_markets_consistent(self):
        matrix = poisson_matrix(1.6, 1.1, -0.05, 10)
        markets = matrix_markets(matrix)
        assert abs(markets[("1X2", "H", None)] + markets[("1X2", "D", None)]
                   + markets[("1X2", "A", None)] - 1.0) < 1e-6
        assert abs(markets[("BTTS", "YES", None)]
                   + markets[("BTTS", "NO", None)] - 1.0) < 1e-6
        # le favori domicile doit etre plus probable
        assert markets[("1X2", "H", None)] > markets[("1X2", "A", None)]
        # double chance = somme des resultats
        assert markets[("DC", "1X", None)] == pytest.approx(
            markets[("1X2", "H", None)] + markets[("1X2", "D", None)])

    def test_rates_from_probs_roundtrip(self):
        lam_h, lam_a = 1.7, 1.2
        matrix = poisson_matrix(lam_h, lam_a, 0.0, 10)
        markets = matrix_markets(matrix)
        h, d, a = (markets[("1X2", "H", None)], markets[("1X2", "D", None)],
                   markets[("1X2", "A", None)])
        rec_h, rec_a = rates_from_probs(h, d, a, markets[("OU", "OVER", 2.5)])
        assert abs(rec_h - lam_h) < 0.15
        assert abs(rec_a - lam_a) < 0.15


class TestStaking:
    def test_kelly_positive_only_with_edge(self):
        assert kelly_stake_fraction(0.55, 2.0, 1.0) > 0
        assert kelly_stake_fraction(0.45, 2.0, 1.0) == 0.0

    def test_flat_stake_capped(self):
        stake = flat_stake(0.6, 3.0, bankroll=1000, kelly_fraction=1.0,
                           max_stake_pct=2.0)
        assert stake == pytest.approx(20.0)  # 2% de 1000

    def test_flat_stake_min(self):
        assert flat_stake(0.4, 2.0, bankroll=10, kelly_fraction=0.25,
                          max_stake_pct=2.0) == 0.0

    def test_edge_and_ev(self):
        assert edge(0.5, 2.1) == pytest.approx(0.05)
        assert expected_value(0.5, 2.1) == pytest.approx(0.05)


def test_selection_labels():
    label = selection_label("1X2", "H", None, "Paris", "Lyon")
    assert "Paris" in label
    assert selection_label("OU", "OVER", 2.5).startswith("Plus de")
    assert "2.5" in selection_label("OU", "UNDER", 2.5)
