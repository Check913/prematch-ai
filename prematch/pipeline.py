# -*- coding: utf-8 -*-
"""Orchestration : ingestion -> features -> prediction -> selection -> digest.

C'est le seul module que le scheduler et le bot Telegram utilisent.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import value
from .config import Config
from .db import Database
from .features import FEATURE_COLUMNS, FeatureBuilder
from .http import PoliteClient
from .logging_setup import get_logger
from .markets import ValuePick
from .model import ModelBundle
from .sources import demo as demo_source
from .sources import football_data, odds_api

LOGGER = get_logger("prematch.pipeline")

DIV_SHORT = {
    "E0": "Premier League", "E1": "Championship", "D1": "Bundesliga",
    "D2": "2. Bundesliga", "F1": "Ligue 1", "F2": "Ligue 2",
    "I1": "Serie A", "I2": "Serie B", "SP1": "Liga", "SP2": "Liga 2",
    "N1": "Eredivisie", "P1": "Liga Portugal", "B1": "Pro League",
    "T1": "Süper Lig", "SC0": "Premiership", "G1": "Super League",
}


class Pipeline:
    """Point d'entree unique pour toutes les operations du systeme."""

    def __init__(self, cfg: Config, db: Database | None = None,
                 client: PoliteClient | None = None):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.db = db or Database(cfg.db_path)
        scraping = cfg.section("scraping")
        self.client = client or PoliteClient(
            cache_dir=cfg.cache_dir,
            timeout=int(scraping.get("timeout_s", 30)),
            min_delay=float(scraping.get("min_delay_s", 2.0)),
            jitter=float(scraping.get("jitter_s", 1.0)),
            max_retries=int(scraping.get("max_retries", 3)),
            respect_robots=bool(scraping.get("respect_robots", True)),
            cache_ttl=int(scraping.get("cache_ttl_s", 900)),
            max_requests_per_hour=int(
                scraping.get("max_requests_per_host_per_hour", 240)))
        self._bundle: ModelBundle | None = None

    # ------------------------------------------------------------- ingestion
    def ingest_history(self) -> dict[str, Any]:
        if self.cfg.enabled("sources.demo"):
            return demo_source.sync_demo(self.db, self.cfg)
        return football_data.sync_history(self.db, self.client, self.cfg)

    def refresh_fixtures(self) -> dict[str, Any]:
        result = football_data.sync_fixtures(self.db, self.client, self.cfg)
        if self.cfg.enabled("sources.odds_api"):
            extra = odds_api.sync(self.db, self.client, self.cfg)
            result = {**result, "odds_api": extra}
        return result

    def ingest_all(self) -> dict[str, Any]:
        return {"history": self.ingest_history(),
                "fixtures": self.refresh_fixtures()}

    # --------------------------------------------------------------- modeles
    def load_bundle(self, force_reload: bool = False) -> ModelBundle:
        """Charge le modele actif (ou le plus recent sur disque)."""
        if self._bundle is not None and not force_reload:
            return self._bundle
        row = self.db.active_model()
        candidates: list[Path] = []
        if row is not None and row["path"]:
            candidates.append(Path(row["path"]))
        candidates.extend(sorted(self.cfg.models_dir.glob("model_*.joblib"),
                                 reverse=True))
        for path in candidates:
            if path.exists():
                self._bundle = ModelBundle.load(path)
                LOGGER.info("modele charge : %s (%s)", path,
                            self._bundle.version)
                return self._bundle
        raise FileNotFoundError(
            "aucun modele entraine : lancez d'abord `prematch train`")

    def train(self, holdout_days: int | None = None) -> dict[str, Any]:
        history = football_data.history_frame(self.db)
        if len(history) < 200:
            raise RuntimeError(
                f"historique insuffisant ({len(history)} matchs) : "
                "lancez `prematch ingest-history` ou le mode demo")
        builder = FeatureBuilder()
        features = builder.build(history)
        days = int(holdout_days if holdout_days is not None
                   else self.cfg.get("learning.holdout_days", 30))
        bundle = ModelBundle.train(history, features, self.cfg,
                                   holdout_days=days)
        path = self.cfg.models_dir / f"model_{bundle.version}.joblib"
        bundle.save(path)
        self.db.register_model(
            bundle.version, "ensemble",
            params={"weight_dc": bundle.weight_dc, "n_train": bundle.n_train},
            metrics=bundle.metrics, path=str(path),
            trained_on=datetime.now(timezone.utc).isoformat(), activate=True)
        self._bundle = bundle
        self._prune_models(keep=5)
        return {"version": bundle.version, "n_train": bundle.n_train,
                "metrics": bundle.metrics, "path": str(path)}

    def _prune_models(self, keep: int = 5) -> None:
        models = sorted(self.cfg.models_dir.glob("model_*.joblib"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        for old in models[keep:]:
            try:
                old.unlink()
            except OSError:  # pragma: no cover
                pass

    # ------------------------------------------------------------- prediction
    def upcoming_features(self, days_ahead: int = 3) -> pd.DataFrame:
        history = football_data.history_frame(self.db)
        fixtures = football_data.upcoming_frame(self.db, days_ahead)
        if fixtures.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        builder = FeatureBuilder()
        combined = builder.build(history, fixtures)
        upcoming = combined[combined["y_result"].isna()].copy()
        return upcoming.reset_index(drop=True)

    def kickoff_map(self, days_ahead: int = 3) -> dict[str, str]:
        fixtures = football_data.upcoming_frame(self.db, days_ahead)
        if fixtures.empty:
            return {}
        return {row["key"]: f"{row['date']}T{row.get('time') or '12:00'}"
                for row in fixtures.to_dict("records")}

    def predict(self, days_ahead: int = 3, store: bool = True
                ) -> dict[str, Any]:
        """Genere les analyses et les paris du jour."""
        bundle = self.load_bundle()
        rows = self.upcoming_features(days_ahead)
        if rows.empty:
            return {"picks": [], "analyses": [], "model_version":
                    bundle.version, "n_matches": 0, "bankroll": self.bankroll()}

        kickoffs = self.kickoff_map(days_ahead)
        results = bundle.market_probabilities(rows)
        for item in results:
            item["kickoff"] = kickoffs.get(item["key"], item["date"])

        odds_by_match = {item["key"]: self.db.latest_odds(item["key"])
                         for item in results}
        method = str(self.cfg.get("model.devig_method", "shin"))
        analyses = [value.analyse_match(item, odds_by_match[item["key"]],
                                        method) for item in results]
        picks = value.build_picks(results, odds_by_match, self.cfg,
                                  self.bankroll(), method)
        if store and picks:
            self.db.save_predictions(
                value.picks_to_rows(picks, bundle.version))
        LOGGER.info("%s matchs analyses, %s paris retenus", len(results),
                    len(picks))
        return {"picks": picks, "analyses": analyses,
                "model_version": bundle.version, "n_matches": len(results),
                "bankroll": self.bankroll()}

    # -------------------------------------------------------------- bankroll
    def bankroll(self) -> float:
        stored = self.db.get_meta("bankroll")
        if isinstance(stored, (int, float)) and stored > 0:
            return float(stored)
        return float(self.cfg.get("app.bankroll_eur", 1000.0))

    def set_bankroll(self, amount: float) -> None:
        self.db.set_meta("bankroll", round(float(amount), 2))

    # ------------------------------------------------------------- reglement
    def settle(self) -> dict[str, Any]:
        settled = self.db.settle_open_bets()
        stats = self.db.bet_stats()
        if settled:
            LOGGER.info("%s paris regles", settled)
            self.db.log_event("settle", f"{settled} paris regles")
        return {"settled": settled, "stats": stats}

    # ---------------------------------------------------------------- digest
    def digest_text(self, result: dict[str, Any] | None = None,
                    title: str | None = None) -> str:
        """Message Telegram (HTML) avec les paris et leurs alternatives."""
        result = result or self.predict(store=False)
        picks: list[ValuePick] = result["picks"]
        analyses = {a["key"]: a for a in result.get("analyses", [])}
        bankroll = float(result.get("bankroll", self.bankroll()))
        today = datetime.now(timezone.utc).strftime("%d/%m/%Y")

        lines: list[str] = []
        lines.append(f"<b>⚽ PREMATCH AI — {html.escape(title or today)}</b>")
        lines.append(f"Modèle <code>{html.escape(result['model_version'])}</code>"
                     f" · bankroll <b>{bankroll:.0f} €</b>")
        if not picks:
            lines.append("")
            lines.append("Aucune opportunité ne dépasse les filtres de valeur "
                         "aujourd'hui. C'est normal : la majorité des jours "
                         "n'offre pas de value réelle. Pas de pari = pas de "
                         "perte.")
            return "\n".join(lines)

        total_stake = sum(p.stake for p in picks)
        lines.append(f"<i>{len(picks)} sélection(s) · "
                     f"{total_stake:.2f} € engagés "
                     f"({total_stake / bankroll * 100:.1f}% de bankroll)</i>")
        lines.append("")

        for index, pick in enumerate(picks, start=1):
            kickoff = pick.kickoff.replace("T", " · ") if pick.kickoff else ""
            league = DIV_SHORT.get(pick.div, pick.div)
            lines.append(f"<b>{index}. {html.escape(pick.home)} – "
                         f"{html.escape(pick.away)}</b>")
            lines.append(f"{html.escape(str(league))} · {html.escape(kickoff)}")
            lines.append(f"🎯 <b>{html.escape(pick.label)}</b> "
                         f"@ <b>{pick.price:.2f}</b> ({html.escape(pick.book)})")
            lines.append(f"Modèle {pick.p_model * 100:.1f}% vs marché "
                         f"{pick.p_market * 100:.1f}% → valeur "
                         f"<b>{pick.edge * 100:+.1f}%</b>")
            lines.append(f"Mise <b>{pick.stake:.2f} €</b> · confiance "
                         f"{pick.confidence * 100:.0f}%")

            analysis = analyses.get(pick.match_key)
            alternatives: list[str] = []
            if analysis:
                for edge in analysis["edges"]:
                    same = (edge["market"], edge["selection"],
                            edge["line"]) == (pick.market, pick.selection,
                                              pick.line)
                    if same or edge["edge"] <= 0.005 or len(alternatives) >= 2:
                        continue
                    label = value.label_for(
                        ValuePick(match_key=pick.match_key, home=pick.home,
                                  away=pick.away, div=pick.div,
                                  kickoff=pick.kickoff, market=edge["market"],
                                  selection=edge["selection"],
                                  line=edge["line"], price=edge["price"],
                                  book=edge["book"], p_model=edge["p_model"],
                                  p_market=edge["p_market"], edge=edge["edge"],
                                  kelly=0.0, stake=0.0, confidence=0.0))
                    alternatives.append(f"{html.escape(label)} @ "
                                        f"{edge['price']:.2f} "
                                        f"({edge['edge'] * 100:+.1f}%)")
            if alternatives:
                lines.append("Autres options : " + " · ".join(alternatives))
            lines.append("")

        lines.append(f"<b>Total :</b> {total_stake:.2f} € "
                     f"({total_stake / bankroll * 100:.1f}% de bankroll)")
        lines.append("<i>Value betting = espérance positive sur la durée, "
                     "pas une certitude par pari. Misez uniquement ce que "
                     "vous pouvez vous permettre de perdre.</i>")
        return "\n".join(lines)

    def stats_text(self) -> str:
        stats = self.db.bet_stats()
        model = self.db.active_model()
        lines = ["<b>📊 Performance</b>"]
        lines.append(f"Paris réglés : {stats['n']} · ROI "
                     f"<b>{stats['roi']:+.2f}%</b> · "
                     f"P&L <b>{stats['pnl']:+.2f} €</b>")
        for status, info in sorted(stats.get("by_status", {}).items()):
            if status in ("won", "lost", "void") and info:
                lines.append(f"· {status} : {info['n']}")
        if model is not None:
            lines.append("")
            lines.append(f"<b>Modèle actif</b> <code>{model['version']}</code>")
            metrics = {}
            try:
                import json
                metrics = json.loads(model["metrics"] or "{}")
            except ValueError:
                pass
            if metrics:
                lines.append(
                    f"LogLoss {metrics.get('holdout_logloss', '?')} · "
                    f"Brier {metrics.get('holdout_brier', '?')} · "
                    f"accuracy {metrics.get('holdout_accuracy', '?')} "
                    f"sur {metrics.get('holdout_n', '?')} matchs")
        return "\n".join(lines)
