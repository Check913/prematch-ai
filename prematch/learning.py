# -*- coding: utf-8 -*-
"""Boucle d'auto-amelioration : re-entrainement, promotion, detection de drift.

Cycle complet (lance automatiquement par le scheduler) :
  1. les matchs termines sont regles (bets -> won/lost) ;
  2. si assez de temps a passe depuis le dernier entrainement, un nouveau
     modele est entraine sur tout l'historique et enregistre ;
  3. le nouveau modele doit battre l'ancien sur le holdout le plus recent
     (logloss) pour etre promu modele actif ;
  4. on mesure la derive du marche (erreur moyenne modele vs marche) : si le
     modele se degrade, la prudence augmente automatiquement (seuil d'edge
     plus strict).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from .config import Config
from .db import Database
from .logging_setup import get_logger
from .pipeline import Pipeline

LOGGER = get_logger("prematch.learning")


def _logloss_from_bets(db: Database) -> float | None:
    """Logloss empirique des predictions utilisees (proxie de calibration)."""
    rows = list(db.conn.execute(
        """SELECT p.p_model, b.status FROM bets b
           JOIN predictions p ON p.id=b.prediction_id
           WHERE b.status IN ('won','lost') AND b.settled_at IS NOT NULL"""))
    if len(rows) < 30:
        return None
    losses = []
    for row in rows:
        p = max(1e-6, min(1 - 1e-6, float(row["p_model"])))
        y = 1.0 if row["status"] == "won" else 0.0
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return sum(losses) / len(losses)


def market_drift(db: Database) -> float | None:
    """Ecart moyen modele / marche sur les derniers paris regles.

    Une derive croissante signifie que le modele s'ecarte de plus en plus du
    marche : signal de degradation (ou d'opportunite, d'ou une reponse
    prudente : durcir le seuil).
    """
    rows = list(db.conn.execute(
        """SELECT p.p_model, p.p_market FROM predictions p
           JOIN bets b ON b.prediction_id=p.id
           WHERE b.status IN ('won','lost') AND b.settled_at IS NOT NULL
           ORDER BY b.settled_at DESC LIMIT 100"""))
    if len(rows) < 20:
        return None
    return sum(abs(float(r["p_model"]) - float(r["p_market"]))
               for r in rows) / len(rows)


def promote_model(db: Database, cfg: Config, candidate: dict[str, Any],
                  incumbent: dict[str, Any] | None) -> bool:
    """Promotion uniquement si le nouveau modele est meilleur sur holdout."""
    learning_cfg = cfg.section("learning")
    gain = float(learning_cfg.get("promo_min_logloss_gain", 0.002))
    new_ll = candidate.get("metrics", {}).get("holdout_logloss")
    if new_ll is None:
        # pas de holdout comparable : on promet (premier modele)
        db.activate_model(candidate["version"])
        db.log_event("model_promoted", f"{candidate['version']} (premier)")
        return True
    if incumbent is None:
        db.activate_model(candidate["version"])
        db.log_event("model_promoted", f"{candidate['version']} (aucun actif)")
        return True
    old_ll = incumbent.get("metrics", {}).get("holdout_logloss")
    if old_ll is None or new_ll < float(old_ll) - gain:
        db.activate_model(candidate["version"])
        db.log_event("model_promoted",
                     f"{candidate['version']} (logloss {new_ll} < {old_ll})")
        return True
    db.log_event("model_rejected",
                 f"{candidate['version']} pas mieux ({new_ll} vs {old_ll})")
    return False


def adjust_risk(db: Database, cfg: Config) -> dict[str, Any]:
    """Adapte le seuil d'edge selon la derive observee."""
    base_edge = float(cfg.get("app.min_edge", 0.03))
    drift = market_drift(db)
    if drift is None:
        return {"min_edge": base_edge, "drift": None}
    # derive < 0.05 : normal ; au-dela, on durcit progressivement
    extra = min(0.03, max(0.0, (drift - 0.05) * 0.6))
    adjusted = round(base_edge + extra, 4)
    db.set_meta("effective_min_edge", adjusted)
    db.log_event("risk_adjusted",
                 f"derive={drift:.4f} -> min_edge={adjusted}")
    return {"min_edge": adjusted, "drift": round(drift, 4)}


def daily_cycle(pipeline: Pipeline) -> dict[str, Any]:
    """Le cycle qui fait que l'agent s'ameliore tout seul, chaque nuit."""
    db = pipeline.db
    cfg = pipeline.cfg
    outcome: dict[str, Any] = {}

    # 1. reglement des paris
    settled = db.settle_open_bets()
    outcome["settled"] = settled

    # 1bis. alerte de perte (non bloquante : le journal + Telegram via events)
    initial = float(cfg.get("app.bankroll_eur", 1000.0))
    stop_pct = float(cfg.get("app.stop_loss_pct", 8.0))
    current = pipeline.bankroll()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if initial > 0 and current < initial * (1 - stop_pct / 100):
        if db.get_meta("last_stop_loss_alert") != today:
            db.log_event(
                "stop_loss_warning",
                f"bankroll {current:.2f} EUR vs initiale {initial:.2f} "
                f"(-{(1 - current / initial) * 100:.1f}% >= seuil {stop_pct}%)",
                level="WARNING")
            db.set_meta("last_stop_loss_alert", today)

    # 2. faut-il re-entrainer ?
    learning_cfg = cfg.section("learning")
    last_train = db.get_meta("last_train_at")
    min_days = int(learning_cfg.get("min_days_between_retrains", 3))
    retrain_due = True
    if last_train:
        try:
            last = datetime.fromisoformat(str(last_train))
            age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
            retrain_due = age_days >= min_days
        except ValueError:
            retrain_due = True

    if retrain_due:
        try:
            trained = pipeline.train()
            outcome["trained"] = trained
            db.set_meta("last_train_at", datetime.now(timezone.utc).isoformat())
            incumbent = None
            # promotion : le train() a deja active le nouveau ; on verifie
            # contre l'ancien enregistre juste avant (registre complet)
            versions = db.model_versions(limit=2)
            if len(versions) == 2:
                incumbent = versions[1]
                promote_model(db, cfg, versions[0], incumbent)
            outcome["promotion"] = True
        except Exception as exc:
            LOGGER.exception("re-entrainement impossible : %s", exc)
            outcome["trained_error"] = str(exc)
            db.log_event("train_error", str(exc), level="ERROR")

    # 3. adaptation du risque
    outcome["risk"] = adjust_risk(db, cfg)

    # 4. metriques de la boucle
    ll = _logloss_from_bets(db)
    if ll is not None:
        db.set_meta("empirical_logloss", round(ll, 5))
        outcome["empirical_logloss"] = round(ll, 5)
    return outcome


def retrain_if_needed(pipeline: Pipeline) -> dict[str, Any]:
    """Commodite : force le cycle (utilise par le scheduler)."""
    return daily_cycle(pipeline)


def learning_summary(db: Database) -> str:
    """Texte Telegram : ce que l'agent a appris recemment."""
    versions = db.model_versions(limit=5)
    lines = ["<b>🧠 Apprentissage</b>"]
    if not versions:
        lines.append("Aucun modele enregistre pour l'instant.")
        return "\n".join(lines)
    for row in versions[:5]:
        active = " ✅" if row["is_active"] else ""
        metrics = {}
        try:
            metrics = json.loads(row["metrics"] or "{}")
        except ValueError:
            pass
        lines.append(
            f"· <code>{row['version']}</code>{active} — "
            f"{row.get('n_matches') or '?'} matchs, logloss "
            f"{metrics.get('holdout_logloss', '?')}")
    drift = market_drift(db)
    if drift is not None:
        lines.append(f"Dérive modèle/marché : {drift:.4f}")
    return "\n".join(lines)
