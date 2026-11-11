# -*- coding: utf-8 -*-
"""Scheduler continu : le cerveau qui tourne 24/7.

Trois taches periodiques (threads) :
  * ingestion des fixtures + cotes (toutes les N minutes) ;
  * reglement des paris + re-entrainement + adaptation du risque (toutes
    les heures, avec verrou) ;
  * entraînement complet de nuit si besoin.

Les heures calmes (nuit) evitent les envois Telegram la nuit.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from .config import Config
from .learning import daily_cycle
from .logging_setup import get_logger
from .pipeline import Pipeline

LOGGER = get_logger("prematch.scheduler")


def in_quiet_hours(cfg: Config, now: datetime | None = None) -> bool:
    start, end = [int(x) for x in cfg.section("scheduler").get(
        "quiet_hours", [1, 7])]
    hour = (now or datetime.now()).hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


class Scheduler:
    """Ordonnanceur simple base sur des threads (pas de dependance cron)."""

    def __init__(self, pipeline: Pipeline, cfg: Config):
        self.pipeline = pipeline
        self.cfg = cfg
        self._stop = threading.Event()
        self.threads: list[threading.Thread] = []

    # ------------------------------------------------------------- taches
    def _refresh_loop(self, interval_minutes: int) -> None:
        interval = max(300, interval_minutes * 60)
        while not self._stop.wait(interval):
            try:
                result = self.pipeline.refresh_fixtures()
                LOGGER.info("fixtures rafraichies : %s", result)
            except Exception as exc:
                LOGGER.warning("refresh fixtures : %s", exc)

    def _learning_loop(self, interval_minutes: int) -> None:
        interval = max(600, interval_minutes * 60)
        while not self._stop.wait(interval):
            if in_quiet_hours(self.cfg):
                continue
            try:
                outcome = daily_cycle(self.pipeline)
                LOGGER.info("cycle apprentissage : %s", outcome)
            except Exception as exc:
                LOGGER.exception("cycle apprentissage : %s", exc)

    # -------------------------------------------------------------- demarrage
    def start(self) -> None:
        refresh_minutes = int(self.cfg.get("scheduler.refresh_minutes", 30))
        settle_minutes = int(self.cfg.get("scheduler.settle_minutes", 60))
        self.threads = [
            threading.Thread(target=self._refresh_loop,
                             args=(refresh_minutes,), daemon=True,
                             name="refresh"),
            threading.Thread(target=self._learning_loop,
                             args=(settle_minutes,), daemon=True,
                             name="learning"),
        ]
        for thread in self.threads:
            thread.start()
        LOGGER.info("scheduler demarre (refresh=%s min, cycle=%s min)",
                    refresh_minutes, settle_minutes)

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(60)
        except KeyboardInterrupt:
            LOGGER.info("arret demandé")
        finally:
            self.stop()
