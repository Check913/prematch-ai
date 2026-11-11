# -*- coding: utf-8 -*-
"""Diagnostic : le modele bat-il reellement le marche, hors echantillon ?

Compare sur les 200 derniers matchs (jamais vus a l'entrainement) :
  * logloss du modele (DC seul, GBM seul, blend) ;
  * logloss du marche (cotes de fermeture deveigees, Shin) ;
  * calibration : probabilite moyenne predite vs frequence observee.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from prematch.config import load_config
from prematch.features import FeatureBuilder
from prematch.logging_setup import setup_logging
from prematch.model import ModelBundle, _blend, _sharpen
from prematch.pipeline import Pipeline
from prematch.sources import football_data as fd
from prematch.value import market_probabilities


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def main() -> int:
    setup_logging("WARNING")
    cfg = load_config(None)
    pipeline = Pipeline(cfg)
    db = pipeline.db

    rows = [dict(r) for r in db.conn.execute(
        """SELECT m.* FROM matches m
           WHERE m.fthg IS NOT NULL
             AND EXISTS (SELECT 1 FROM odds o WHERE o.match_key=m.key
                         AND o.is_closing=1)
           ORDER BY m.date DESC LIMIT 200""")]
    frame = pd.DataFrame(rows)
    recent_keys = set(frame["key"])
    hist = fd.history_frame(db)
    hist_only = hist[~hist["key"].isin(recent_keys)].reset_index(drop=True)

    builder = FeatureBuilder()
    features_train = builder.build(hist_only)
    # holdout interne 30 j : sert aussi a mesurer la temperature du blend
    bundle = ModelBundle.train(hist_only, features_train, cfg, holdout_days=30)
    print(f"modele entraine sur {len(hist_only)} matchs "
          f"(poids DC={bundle.weight_dc:.2f}, "
          f"T={bundle.temperature:.2f}, "
          f"logloss holdout={bundle.metrics.get('holdout_logloss_sharp', '?')})")

    features_eval = builder.build(hist_only, frame)
    eval_rows = features_eval[
        features_eval["key"].isin(recent_keys)
        & features_eval["y_result"].notna()].drop_duplicates(
        subset=["key"], keep="first")
    y = eval_rows["y_result"].astype(int).to_numpy()

    p_dc = np.vstack([bundle.dc.proba_1x2(r.home, r.away, r.div)
                      for r in eval_rows.itertuples(index=False)])
    p_gbm = bundle.gbm_1x2.predict_proba(eval_rows[list(
        bundle.feature_names)])
    p_blend = _blend(p_dc, p_gbm, bundle.weight_dc)

    # probabilites du marche (fermeture, deveigees)
    p_market = []
    for key in eval_rows["key"]:
        odds = db.closing_odds(key)
        probs = market_probabilities(odds, "shin")
        row = [probs.get(("1X2", "H", None), np.nan),
               probs.get(("1X2", "D", None), np.nan),
               probs.get(("1X2", "A", None), np.nan)]
        p_market.append(row)
    p_market = np.array(p_market, dtype=float)
    ok = ~np.isnan(p_market).any(axis=1)

    print(f"\nMatchs avec cotes : {ok.sum()}/{len(y)}")
    print(f"logloss DC seul    : {logloss(y[ok], p_dc[ok]):.4f}")
    print(f"logloss GBM seul   : {logloss(y[ok], p_gbm[ok]):.4f}")
    print(f"logloss blend      : {logloss(y[ok], p_blend[ok]):.4f}")
    p_sharp = _sharpen(p_blend, bundle.temperature)
    print(f"logloss blend T={bundle.temperature:.2f} : "
          f"{logloss(y[ok], p_sharp):.4f}")
    print(f"logloss MARCHE     : {logloss(y[ok], p_market[ok]):.4f}")

    print("\nCalibration (proba moyenne predite vs frequence reelle) :")
    for name, probs in (("DC", p_dc), ("GBM", p_gbm), ("blend", p_blend),
                        ("marche", p_market)):
        means = probs[ok].mean(axis=0)
        freqs = np.bincount(y[ok], minlength=3) / ok.sum()
        print(f"  {name:6s} H {means[0]:.3f}/{freqs[0]:.3f}  "
              f"N {means[1]:.3f}/{freqs[1]:.3f}  "
              f"A {means[2]:.3f}/{freqs[2]:.3f}")

    # temperature : T<1 = le modele doit etre plus tranchant
    from scipy.optimize import minimize_scalar
    def ll_for_temp(t):
        sharp = np.power(np.clip(p_blend[ok], 1e-9, 1), 1.0 / t)
        sharp = sharp / sharp.sum(axis=1, keepdims=True)
        return logloss(y[ok], sharp)
    res = minimize_scalar(ll_for_temp, bounds=(0.3, 3.0), method="bounded")
    print(f"\nTemperature optimale du blend : T={res.x:.3f} "
          f"(logloss {res.fun:.4f} vs {ll_for_temp(1.0):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
