# -*- coding: utf-8 -*-
"""Chargement de la configuration (YAML + .env + variables d'environnement).

Ordre de priorite (du plus fort au plus faible) :
  1. variables d'environnement du processus ;
  2. fichier .env a la racine du projet ;
  3. config/config.yaml ;
  4. valeurs par defaut ci-dessous.

Les secrets (token Telegram, cle The Odds API) ne sont jamais ecrits dans
config.yaml : on ne stocke que le *nom* de la variable d'environnement.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Valeurs par defaut : le YAML ne ecrase que ce qu'il declare.
DEFAULTS: dict[str, Any] = {
    "app": {
        "timezone": "Europe/Paris",
        "data_dir": "data",
        "log_level": "INFO",
        "bankroll_eur": 1000.0,
        "kelly_fraction": 0.25,
        "max_stake_pct": 2.0,
        "daily_max_exposure_pct": 10.0,
        "stop_loss_pct": 8.0,
        "min_edge": 0.03,
        "max_picks_per_day": 8,
        "min_price": 1.40,
        "max_price": 8.0,
        "min_model_prob": 0.15,
        "min_agreement": 0.55,
    },
    "sources": {
        "football_data": {
            "enabled": True,
            "base_url": "https://www.football-data.co.uk",
            "divisions": [
                "E0", "E1", "D1", "D2", "F1", "F2", "I1", "I2",
                "SP1", "SP2", "N1", "B1", "P1", "T1", "G1", "SC0",
            ],
            "seasons": ["2324", "2425", "2526", "2627"],
            "history_ttl_s": 86400,
            "fixtures_ttl_s": 900,
        },
        "odds_api": {
            "enabled": False,
            "api_key_env": "ODDS_API_KEY",
            "regions": "eu",
            "markets": "h2h,totals",
            "base_url": "https://api.the-odds-api.com/v4",
            "sport_keys": ["soccer_epl", "soccer_spain_la_liga",
                           "soccer_france_ligue_one", "soccer_germany_bundesliga",
                           "soccer_italy_serie_a"],
        },
        "demo": {"enabled": False, "matches": 600, "seed": 42},
    },
    "scraping": {
        "min_delay_s": 2.0,
        "jitter_s": 1.0,
        "max_retries": 3,
        "timeout_s": 30,
        "respect_robots": True,
        "cache_ttl_s": 900,
        "max_requests_per_host_per_hour": 240,
    },
    "model": {
        "dc_decay_per_day": 0.0018,
        "max_goals": 10,
        "min_matches_per_team": 3,
        "gbm_estimators": 250,
        "gbm_learning_rate": 0.05,
        "gbm_max_depth": 3,
        "gbm_min_samples_leaf": 25,
        "calibrate": True,
        "ensemble_weights": {"dc": 0.5, "gbm": 0.5},
        "devig_method": "shin",
    },
    "learning": {
        "retrain_hour": 4,
        "min_days_between_retrains": 3,
        "promo_min_logloss_gain": 0.002,
        "holdout_days": 30,
        "tune_min_edge_grid": [0.02, 0.03, 0.04, 0.05, 0.06],
    },
    "telegram": {
        "enabled": True,
        "token_env": "TELEGRAM_BOT_TOKEN",
        "chat_id_env": "TELEGRAM_CHAT_ID",
        "digest_hours": [9, 17],
        "alert_new_signals": True,
        "max_picks_per_message": 6,
    },
    "scheduler": {
        "refresh_minutes": 30,
        "settle_minutes": 60,
        "quiet_hours": [1, 7],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusion recursive : les dicts sont fusionnes, le reste est remplace."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_env_file(path: Path) -> None:
    """Charge un .env simple (KEY=VALUE) sans ecraser l'environnement."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class Config:
    """Acces pratique a la configuration : cfg.get("app.bankroll_eur")."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        self.data = data
        self.root = Path(root)

    # ------------------------------------------------------------------ acces
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    def enabled(self, path: str) -> bool:
        return bool(self.get(f"{path}.enabled", False))

    # ------------------------------------------------------------------ chemins
    @property
    def data_dir(self) -> Path:
        path = Path(str(self.get("app.data_dir", "data")))
        return path if path.is_absolute() else self.root / path

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "prematch.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.cache_dir, self.models_dir,
                     self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ secrets
    def secret(self, env_name: str, default: str | None = None) -> str | None:
        value = os.environ.get(env_name)
        return value if value else default

    def telegram_token(self) -> str | None:
        return self.secret(str(self.get("telegram.token_env",
                                        "TELEGRAM_BOT_TOKEN")))

    def telegram_chat_id(self) -> str | None:
        return self.secret(str(self.get("telegram.chat_id_env",
                                        "TELEGRAM_CHAT_ID")))


def load_config(path: str | Path | None = None,
                root: Path = PROJECT_ROOT) -> Config:
    """Charge la configuration finale (defauts + YAML + .env)."""
    load_env_file(root / ".env")
    config_path = Path(path) if path else root / "config" / "config.yaml"
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config invalide (dict attendu) : {config_path}")
        data = loaded
    merged = _deep_merge(DEFAULTS, data)
    # Surcharges d'environnement utiles en production / CI.
    for env_key, cfg_path, cast in (
        ("PREMATCH_DATA_DIR", "app.data_dir", str),
        ("PREMATCH_LOG_LEVEL", "app.log_level", str),
        ("PREMATCH_BANKROLL", "app.bankroll_eur", float),
        ("PREMATCH_MIN_EDGE", "app.min_edge", float),
    ):
        raw = os.environ.get(env_key)
        if raw:
            try:
                merged.setdefault(env_key, None)
                target = merged
                parts = cfg_path.split(".")
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = cast(raw)
            except (TypeError, ValueError):
                pass
    return Config(merged, root=root)
