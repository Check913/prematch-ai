# -*- coding: utf-8 -*-
"""Modeles de prediction.

Trois briques complementaires :

1. `DixonColesModel` : modele de buts (Poisson bivarie avec correction
   Dixon-Coles et ponderation temporelle). Il capte la structure du football
   (force offensive/defensive, avantage du terrain, niveau de la division) et
   fournit une matrice de scores complete -> tous les marches.

2. `GBMModel` : gradient boosting (HistGradientBoosting) sur les variables
   de forme / Elo / xG, calibre par regression isotonique out-of-fold. Il
   capte les interactions non lineaires que le modele de buts ignore.

3. `ModelBundle` : melange des deux en log-probabilite (poids ajuste sur une
   fenetre de validation), puis reconstruction d'une matrice de scores
   coherente avec les probabilites finales 1N2 + Plus/Moins 2.5.

Tout est serialisable (joblib) pour le registre de modeles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from .features import FEATURE_COLUMNS, _to_date, canonical_team
from .logging_setup import get_logger
from .markets import matrix_markets, poisson_matrix, rates_from_probs

LOGGER = get_logger("prematch.model")
EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. Dixon-Coles
# ---------------------------------------------------------------------------


class DixonColesModel:
    """Poisson bivarie pondere dans le temps, par equipe et par division."""

    def __init__(self, decay_per_day: float = 0.0018, max_goals: int = 10,
                 ridge: float = 0.6,
                 rho_grid: tuple[float, ...] = (
                     -0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0,
                     0.03, 0.06, 0.09)):
        self.decay_per_day = float(decay_per_day)
        self.max_goals = int(max_goals)
        self.ridge = float(ridge)
        self.rho_grid = tuple(rho_grid)
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.league_offset: dict[str, float] = {}
        self.home_adv = 0.15
        self.rho = -0.05
        self.teams: list[str] = []
        self.divs: list[str] = []
        self.n_matches = 0

    # ------------------------------------------------------------------ fit
    def _prepare(self, matches: pd.DataFrame):
        frame = matches.copy()
        frame["_date"] = frame["date"].map(_to_date)
        frame = frame.dropna(subset=["fthg", "ftag"])
        frame = frame.sort_values("_date")
        frame["home_key"] = frame["home"].map(canonical_team)
        frame["away_key"] = frame["away"].map(canonical_team)
        frame["div_key"] = frame["div"].astype(str)

        self.teams = sorted(set(frame["home_key"]) | set(frame["away_key"]))
        self.divs = sorted(set(frame["div_key"]))
        team_index = {t: i for i, t in enumerate(self.teams)}
        div_index = {d: i for i, d in enumerate(self.divs)}

        last_day = frame["_date"].max()
        days = np.array([(last_day - d).days for d in frame["_date"]],
                        dtype=float)
        weights = np.exp(-self.decay_per_day * days)

        ih = frame["home_key"].map(team_index).to_numpy(dtype=int)
        ia = frame["away_key"].map(team_index).to_numpy(dtype=int)
        di = frame["div_key"].map(div_index).to_numpy(dtype=int)
        hg = frame["fthg"].to_numpy(dtype=float)
        ag = frame["ftag"].to_numpy(dtype=float)
        return frame, ih, ia, di, hg, ag, weights

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        frame, ih, ia, di, hg, ag, weights = self._prepare(matches)
        n, m = len(self.teams), max(1, len(self.divs))
        self.n_matches = int(len(hg))
        if n == 0 or len(hg) < 30:
            LOGGER.warning("trop peu de matchs (%s) pour Dixon-Coles", len(hg))
            return self

        games = np.zeros(n)
        scored = np.zeros(n)
        conceded = np.zeros(n)
        np.add.at(games, ih, 1.0)
        np.add.at(games, ia, 1.0)
        np.add.at(scored, ih, hg)
        np.add.at(scored, ia, ag)
        np.add.at(conceded, ia, hg)
        np.add.at(conceded, ih, ag)
        games = np.clip(games, 1.0, None)
        base_rate = float(np.mean(scored) / max(1e-6, np.mean(games))) or 1.35

        att0 = np.log(np.clip(scored / games, 0.15, None) / base_rate)
        dfn0 = np.log(np.clip(conceded / games, 0.15, None) / base_rate)
        x0 = np.concatenate([att0, dfn0, np.array([0.25]), np.zeros(m)])
        # Bornes strictes : sans elles, l'optimiseur peut faire diverger
        # l'avantage du terrain et les notes s'annulent via le clipping.
        bounds = ([( -2.0, 2.0)] * n + [(-2.0, 2.0)] * n
                  + [(0.0, 0.8)] + [(-1.0, 1.0)] * m)

        def objective(params: np.ndarray):
            att = params[:n]
            dfn = params[n:2 * n]
            hadv = params[2 * n]
            off = params[2 * n + 1:]
            theta_h = att[ih] + dfn[ia] + hadv + off[di]
            theta_a = att[ia] + dfn[ih] + off[di]
            lam_h = np.exp(np.clip(theta_h, -4.0, 3.0))
            lam_a = np.exp(np.clip(theta_a, -4.0, 3.0))
            loglik = float(np.sum(weights * (hg * theta_h - lam_h
                                             + ag * theta_a - lam_a)))
            reg = 0.5 * self.ridge * (float(np.sum(att ** 2))
                                      + float(np.sum(dfn ** 2)))
            reg += 200.0 * (float(np.mean(att)) ** 2)
            reg += 200.0 * (float(np.mean(off)) ** 2)
            value = loglik - reg

            g_h = weights * (hg - lam_h)
            g_a = weights * (ag - lam_a)
            grad = np.zeros_like(params)
            np.add.at(grad[:n], ih, g_h)
            np.add.at(grad[:n], ia, g_a)
            np.add.at(grad[n:2 * n], ia, g_h)
            np.add.at(grad[n:2 * n], ih, g_a)
            grad[2 * n] = float(np.sum(g_h))
            off_grad = np.zeros(m)
            np.add.at(off_grad, di, g_h + g_a)
            grad[2 * n + 1:] = off_grad
            grad[:n] -= self.ridge * att
            grad[n:2 * n] -= self.ridge * dfn
            grad[:n] += 400.0 * np.mean(att) / n
            grad[2 * n + 1:] += 400.0 * np.mean(off) / m
            return -value, -grad

        try:
            res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 300, "maxfun": 80000})
            params = res.x
        except Exception as exc:  # pragma: no cover - repli numerique
            LOGGER.warning("optimisation Dixon-Coles echouee (%s)", exc)
            params = x0

        att = params[:n]
        dfn = params[n:2 * n]
        self.home_adv = float(params[2 * n])
        off = params[2 * n + 1:]
        self.attack = {t: float(att[i]) for i, t in enumerate(self.teams)}
        self.defence = {t: float(dfn[i]) for i, t in enumerate(self.teams)}
        self.league_offset = {d: float(off[i]) for i, d in enumerate(self.divs)}
        self.rho = self._best_rho(ih, ia, di, hg, ag, weights)
        LOGGER.info("Dixon-Coles : %s matchs, %s equipes, rho=%.3f, "
                    "avantage terrain=%.3f", self.n_matches, n, self.rho,
                    self.home_adv)
        return self

    def _lambdas(self, ih, ia, di):
        att = np.array([self.attack.get(t, 0.0) for t in self.teams])
        dfn = np.array([self.defence.get(t, 0.0) for t in self.teams])
        off = np.array([self.league_offset.get(d, 0.0) for d in self.divs])
        theta_h = att[ih] + dfn[ia] + self.home_adv + off[di]
        theta_a = att[ia] + dfn[ih] + off[di]
        return (np.exp(np.clip(theta_h, -3.0, 2.0)),
                np.exp(np.clip(theta_a, -3.0, 2.0)))

    def _best_rho(self, ih, ia, di, hg, ag, weights) -> float:
        lam_h, lam_a = self._lambdas(ih, ia, di)
        best_rho, best_score = 0.0, -np.inf
        for rho in self.rho_grid:
            tau = np.ones_like(lam_h)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
            tau[m01] = 1.0 + lam_h[m01] * rho
            tau[m10] = 1.0 + lam_a[m10] * rho
            tau[m11] = 1.0 - rho
            if np.any(tau <= 0):
                continue
            score = float(np.sum(weights * np.log(tau)))
            if score > best_score:
                best_rho, best_score = float(rho), score
        return best_rho

    # -------------------------------------------------------------- predict
    def rates(self, home: str, away: str, div: str) -> tuple[float, float]:
        h = canonical_team(home)
        a = canonical_team(away)
        off = self.league_offset.get(str(div), 0.0)
        theta_h = self.attack.get(h, 0.0) + self.defence.get(a, 0.0) \
            + self.home_adv + off
        theta_a = self.attack.get(a, 0.0) + self.defence.get(h, 0.0) + off
        lam_h = math.exp(min(max(theta_h, -2.0), 2.2))
        lam_a = math.exp(min(max(theta_a, -2.0), 2.2))
        return (min(max(lam_h, 0.08), 5.0), min(max(lam_a, 0.08), 5.0))

    def score_matrix(self, home: str, away: str, div: str) -> np.ndarray:
        lam_h, lam_a = self.rates(home, away, div)
        return poisson_matrix(lam_h, lam_a, self.rho, self.max_goals)

    def markets(self, home: str, away: str, div: str) -> dict:
        return matrix_markets(self.score_matrix(home, away, div))

    def proba_1x2(self, home: str, away: str, div: str) -> np.ndarray:
        markets = self.markets(home, away, div)
        return np.array([markets[("1X2", "H", None)],
                         markets[("1X2", "D", None)],
                         markets[("1X2", "A", None)]])


# ---------------------------------------------------------------------------
# 2. Gradient boosting calibre
# ---------------------------------------------------------------------------


class GBMModel:
    """Gradient boosting (1N2 multi-classe ou binaire) avec calibration."""

    def __init__(self, n_classes: int = 3, max_iter: int = 250,
                 learning_rate: float = 0.05, max_depth: int = 3,
                 min_samples_leaf: int = 25, l2_regularization: float = 1.0,
                 calibrate: bool = True, random_state: int = 42):
        self.n_classes = int(n_classes)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.calibrate = bool(calibrate)
        self.random_state = int(random_state)
        self.model: HistGradientBoostingClassifier | None = None
        self.calibrators: list[IsotonicRegression] = []
        self.classes_: list[int] = []
        self.prior_: np.ndarray | None = None

    # -------------------------------------------------------------- interne
    def _new_estimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )

    @staticmethod
    def _align(proba: np.ndarray, classes: list[int],
               target: list[int]) -> np.ndarray:
        out = np.zeros((proba.shape[0], len(target)))
        for j, cls in enumerate(classes):
            out[:, target.index(cls)] = proba[:, j]
        return out

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame | np.ndarray,
            y: pd.Series | np.ndarray) -> "GBMModel":
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y)
        mask = ~pd.isna(y)
        X_fit, y_fit = Xa[mask], ya[mask].astype(int)
        self.classes_ = sorted(np.unique(y_fit).tolist())
        if len(self.classes_) < 2:
            self.prior_ = np.ones(len(self.classes_), dtype=float)
            self.model = None
            return self

        if self.calibrate and len(X_fit) >= 200:
            oof = np.zeros((len(y_fit), len(self.classes_)))
            counts = np.bincount(y_fit, minlength=max(self.classes_) + 1)
            n_splits = int(min(4, max(2, counts[counts > 0].min())))
            try:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=self.random_state)
                for train_idx, val_idx in skf.split(X_fit, y_fit):
                    est = self._new_estimator()
                    est.fit(X_fit[train_idx], y_fit[train_idx])
                    proba = est.predict_proba(X_fit[val_idx])
                    oof[val_idx] = self._align(proba, list(est.classes_),
                                               self.classes_)
                self.calibrators = []
                for k, cls in enumerate(self.classes_):
                    iso = IsotonicRegression(out_of_bounds="clip",
                                             y_min=1e-4, y_max=1 - 1e-4)
                    iso.fit(oof[:, k], (y_fit == cls).astype(int))
                    self.calibrators.append(iso)
            except ValueError as exc:  # pragma: no cover - tres petits jeux
                LOGGER.debug("calibration impossible (%s)", exc)
                self.calibrators = []

        self.model = self._new_estimator()
        self.model.fit(X_fit, y_fit)
        self.prior_ = (np.bincount([self.classes_.index(v) for v in y_fit],
                                   minlength=len(self.classes_))
                       / len(y_fit))
        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        if self.model is None or not self.classes_:
            n = len(self.classes_) or self.n_classes
            return np.tile(np.ones(n) / n, (len(Xa), 1))
        raw = self.model.predict_proba(Xa)
        out = self._align(raw, list(self.model.classes_), self.classes_)
        if self.calibrators:
            calibrated = np.column_stack([
                iso.predict(out[:, k]) for k, iso in enumerate(self.calibrators)
            ])
            calibrated = np.clip(calibrated, 1e-4, None)
            out = calibrated / calibrated.sum(axis=1, keepdims=True)
        return out


# ---------------------------------------------------------------------------
# 3. Ensemble + persistance
# ---------------------------------------------------------------------------


def _blend(p_dc: np.ndarray, p_gbm: np.ndarray, weight_dc: float) -> np.ndarray:
    """Melange geometrique : p ∝ p_dc^w * p_gbm^(1-w)."""
    w = float(min(max(weight_dc, 0.0), 1.0))
    p = np.power(np.clip(p_dc, EPS, 1), w) * np.power(np.clip(p_gbm, EPS, 1),
                                                      1 - w)
    return p / p.sum(axis=-1, keepdims=True)


def _sharpen(p: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrage par temperature : T<1 rend le modele plus tranchant.

    Le ridge et la moyenne d'ensembles aplatissent les probabilites vers
    1/3, ce qui cree des edges fantomes sur les longshots. Mesurer T sur
    un holdout et trancher corrige l'essentiel de ce biais.
    """
    t = float(min(max(temperature, 0.1), 3.0))
    if abs(t - 1.0) < 1e-6:
        return p
    x = np.power(np.clip(p, EPS, 1.0), 1.0 / t)
    return x / x.sum(axis=-1, keepdims=True)


@dataclass
class ModelBundle:
    """Modele pret a l'emploi + metadonnees de version."""

    dc: DixonColesModel
    gbm_1x2: GBMModel
    gbm_over: GBMModel
    weight_dc: float = 0.5
    temperature: float = 1.0
    version: str = ""
    trained_at: str = ""
    n_train: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # -------------------------------------------------------------- entrainement
    @classmethod
    def train(cls, matches: pd.DataFrame, features: pd.DataFrame,
              cfg: Any, holdout_days: int = 30) -> "ModelBundle":
        """Entraine DC + GBM, ajuste le poids de l'ensemble sur le holdout."""
        model_cfg = cfg.section("model")
        labelled = features.dropna(subset=["y_result"]).copy()
        if labelled.empty:
            raise ValueError("aucun match etiquete pour l'entrainement")

        labelled["_date"] = labelled["date"].map(_to_date)
        cutoff = labelled["_date"].max() - pd.Timedelta(days=holdout_days)
        train_mask = labelled["_date"] <= cutoff
        if train_mask.sum() < 200:  # pas assez d'historique : tout en train
            train_mask = pd.Series(True, index=labelled.index)

        dc = DixonColesModel(
            decay_per_day=float(model_cfg.get("dc_decay_per_day", 0.0018)),
            max_goals=int(model_cfg.get("max_goals", 10)))
        dc.fit(matches)

        X = labelled[FEATURE_COLUMNS]
        y = labelled["y_result"].astype(int)
        y_over = labelled["y_over25"].astype(int)

        gbm_1x2 = GBMModel(
            n_classes=3,
            max_iter=int(model_cfg.get("gbm_estimators", 250)),
            learning_rate=float(model_cfg.get("gbm_learning_rate", 0.05)),
            max_depth=int(model_cfg.get("gbm_max_depth", 3)),
            min_samples_leaf=int(model_cfg.get("gbm_min_samples_leaf", 25)),
            calibrate=bool(model_cfg.get("calibrate", True)))
        gbm_over = GBMModel(
            n_classes=2,
            max_iter=int(model_cfg.get("gbm_estimators", 250)),
            learning_rate=float(model_cfg.get("gbm_learning_rate", 0.05)),
            max_depth=int(model_cfg.get("gbm_max_depth", 3)),
            min_samples_leaf=int(model_cfg.get("gbm_min_samples_leaf", 25)),
            calibrate=bool(model_cfg.get("calibrate", True)))

        gbm_1x2.fit(X[train_mask], y[train_mask])
        gbm_over.fit(X[train_mask], y_over[train_mask])

        # Poids de l'ensemble : logloss sur le holdout.
        weight = float(model_cfg.get("ensemble_weights", {}).get("dc", 0.5))
        holdout = labelled[~train_mask]
        metrics: dict[str, Any] = {}
        if len(holdout) >= 50:
            p_dc = np.vstack([
                dc.proba_1x2(row.home, row.away, row.div)
                for row in holdout.itertuples(index=False)])
            p_gbm = gbm_1x2.predict_proba(holdout[FEATURE_COLUMNS])
            y_val = holdout["y_result"].astype(int).to_numpy()
            best_w, best_ll = weight, float("inf")
            for candidate in np.arange(0.0, 1.01, 0.05):
                blended = _blend(p_dc, p_gbm, float(candidate))
                ll = log_loss(y_val, blended, labels=[0, 1, 2])
                if ll < best_ll:
                    best_w, best_ll = float(candidate), float(ll)
            weight = best_w

            blended = _blend(p_dc, p_gbm, weight)

            # Temperature : T<1 = le blend est trop plat (sous-confiance).
            def _ll_temp(t: float) -> float:
                return float(log_loss(y_val, _sharpen(blended, t),
                                      labels=[0, 1, 2]))

            temp_res = minimize_scalar(_ll_temp, bounds=(0.5, 1.5),
                                       method="bounded")
            temperature = float(temp_res.x)
            n_val = len(y_val)
            onehot = np.zeros_like(blended)
            onehot[np.arange(n_val), y_val] = 1.0
            brier = float(np.mean(np.sum((blended - onehot) ** 2, axis=1)))
            metrics = {
                "holdout_n": int(n_val),
                "holdout_logloss": round(best_ll, 5),
                "holdout_logloss_gbm": round(float(log_loss(
                    y_val, p_gbm, labels=[0, 1, 2])), 5),
                "holdout_logloss_dc": round(float(log_loss(
                    y_val, p_dc, labels=[0, 1, 2])), 5),
                "holdout_brier": round(brier, 5),
                "holdout_accuracy": round(float(np.mean(
                    blended.argmax(axis=1) == y_val)), 4),
                "weight_dc": round(weight, 3),
                "temperature": round(temperature, 3),
                "holdout_logloss_sharp": round(_ll_temp(temperature), 5),
            }

        # Reajustement final sur toutes les donnees (le holdout a servi a
        # choisir le poids ; le modele final voit tout l'historique).
        gbm_1x2.fit(X, y)
        gbm_over.fit(X, y_over)

        version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
        bundle = cls(dc=dc, gbm_1x2=gbm_1x2, gbm_over=gbm_over,
                     weight_dc=weight,
                     temperature=float(metrics.get("temperature", 1.0)),
                     version=version,
                     trained_at=datetime.now(timezone.utc).isoformat(),
                     n_train=int(len(labelled)), metrics=metrics)
        LOGGER.info("Modele %s entraine sur %s matchs (poids DC=%.2f, "
                    "T=%.2f)", version, len(labelled), weight,
                    bundle.temperature)
        return bundle

    # ------------------------------------------------------------------ usage
    def proba_1x2_for(self, home: str, away: str, div: str,
                      features_row: pd.Series | dict) -> np.ndarray:
        p_dc = self.dc.proba_1x2(home, away, div)
        x = pd.DataFrame([{name: features_row.get(name, 0.0)
                           for name in FEATURE_COLUMNS}])
        p_gbm = self.gbm_1x2.predict_proba(x)[0]
        return _sharpen(_blend(p_dc, p_gbm, self.weight_dc),
                        getattr(self, "temperature", 1.0))

    def market_probabilities(self, rows: pd.DataFrame) -> list[dict]:
        """Probabilites de tous les marches pour chaque ligne de features."""
        if rows.empty:
            return []
        X = rows[FEATURE_COLUMNS]
        p_gbm = self.gbm_1x2.predict_proba(X)
        p_over = self.gbm_over.predict_proba(X)[:, 1] \
            if self.gbm_over.classes_ and len(self.gbm_over.classes_) > 1 \
            else np.full(len(rows), 0.5)

        temperature = float(getattr(self, "temperature", 1.0))
        results: list[dict] = []
        for i, row in enumerate(rows.itertuples(index=False)):
            p_dc = self.dc.proba_1x2(row.home, row.away, row.div)
            p = _sharpen(_blend(p_dc, p_gbm[i], self.weight_dc), temperature)
            lam_h, lam_a = rates_from_probs(float(p[0]), float(p[1]),
                                            float(p[2]), float(p_over[i]))
            matrix = poisson_matrix(lam_h, lam_a, self.dc.rho, self.dc.max_goals)
            markets = matrix_markets(matrix)
            # coherence : on impose les probabilites de l'ensemble
            markets[("1X2", "H", None)] = float(p[0])
            markets[("1X2", "D", None)] = float(p[1])
            markets[("1X2", "A", None)] = float(p[2])
            markets[("DC", "1X", None)] = float(p[0] + p[1])
            markets[("DC", "12", None)] = float(p[0] + p[2])
            markets[("DC", "X2", None)] = float(p[1] + p[2])
            markets[("OU", "OVER", 2.5)] = float(p_over[i])
            markets[("OU", "UNDER", 2.5)] = float(1.0 - p_over[i])
            results.append({
                "key": row.key, "home": row.home, "away": row.away,
                "div": row.div, "date": row.date,
                "p_1x2": p, "p_dc": p_dc, "p_gbm": p_gbm[i],
                "p_over25": float(p_over[i]),
                "lam_home": lam_h, "lam_away": lam_a,
                "markets": markets,
            })
        return results

    # ------------------------------------------------------------- persistance
    def save(self, path: str | Path) -> str:
        path = str(path)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        bundle = joblib.load(str(path))
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"objet inattendu dans {path}")
        return bundle
