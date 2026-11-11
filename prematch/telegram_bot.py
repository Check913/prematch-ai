# -*- coding: utf-8 -*-
"""Bot Telegram : digest quotidien, alertes de value, commandes et panneau
de controle par boutons.

Fonctions :
  * digest aux heures configurees (paris du jour, mise Kelly, alternatives) ;
  * alerte immediate quand un nouveau signal depasse un seuil renforce
    (edge >= 2x le seuil de base) ;
  * commandes : /start, /today, /stats, /learning, /bankroll N, /menu,
    /pause, /resume, /edge X, /help ;
  * PANNEAU DE CONTROLE par boutons inline : digest immediat, rafraichir
    les cotes, entrainement a la demande, pause/reprise des notifications,
    reglage du seuil d'edge ;
  * SECURITE : seuls les messages/callbacks du chat configure
    (TELEGRAM_CHAT_ID) sont obey — le bot ignore poliment tout le reste.

Le token et le chat id viennent de l'environnement (jamais en clair).
"""
from __future__ import annotations

import html
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .config import Config
from .logging_setup import get_logger
from .pipeline import Pipeline

LOGGER = get_logger("prematch.telegram")

API = "https://api.telegram.org/bot{token}/{method}"

EDGE_PRESETS = [0.02, 0.03, 0.04, 0.05, 0.08, 0.10]

BOT_COMMANDS = [
    {"command": "today", "description": "Les paris du jour"},
    {"command": "stats", "description": "Performance (ROI, P&L)"},
    {"command": "learning", "description": "Ce que l'IA a appris"},
    {"command": "menu", "description": "Panneau de controle (boutons)"},
    {"command": "pause", "description": "Couper les notifications"},
    {"command": "resume", "description": "Reprendre les notifications"},
    {"command": "edge", "description": "Seuil d'edge, ex : /edge 0.04"},
    {"command": "bankroll", "description": "Bankroll, ex : /bankroll 1000"},
    {"command": "help", "description": "Aide"},
]


class TelegramClient:
    """Client minimal (pas de dependance lourde)."""

    def __init__(self, token: str, chat_id: str | None = None,
                 timeout: int = 30):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.session = requests.Session()

    def call(self, method: str, payload: dict[str, Any]) -> dict:
        url = API.format(token=self.token, method=method)
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        data = resp.json()
        if not data.get("ok"):
            LOGGER.warning("Telegram %s -> %s", method, data)
            raise RuntimeError(f"telegram {method}: {data.get('description')}")
        return data["result"]

    def send(self, text: str, chat_id: str | None = None,
             keyboard: list[list[dict]] | None = None) -> int | None:
        payload: dict[str, Any] = {
            "chat_id": chat_id or self.chat_id,
            "text": text[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        result = self.call("sendMessage", payload)
        return result.get("message_id")

    def register_commands(self, commands: list[dict[str, str]]) -> None:
        """Enregistre les commandes dans le menu officiel du bot."""
        try:
            self.call("setMyCommands", {"commands": commands})
        except RuntimeError as exc:
            LOGGER.warning("setMyCommands : %s", exc)


def split_message(text: str, limit: int = 3800) -> list[str]:
    """Decoupe un long texte en morceaux valides pour Telegram."""
    chunks: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = (current + "\n\n" + block).strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [""]


def build_keyboard(callback_prefix: str = "pick") -> list[list[dict]]:
    return [[{"text": "📊 Stats", "callback_data": "stats"},
             {"text": "🔄 Aujourd'hui", "callback_data": "today"},
             {"text": "⚙️ Menu", "callback_data": "menu"}]]


class TelegramBot:
    """Boucle longue : polling + envoi planifie des digests + panneau."""

    def __init__(self, pipeline: Pipeline, cfg: Config):
        self.pipeline = pipeline
        self.cfg = cfg
        token = cfg.telegram_token()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN absent de l'environnement")
        self.client = TelegramClient(token, cfg.telegram_chat_id())
        self.authorized_chat = str(cfg.telegram_chat_id() or "")
        self.tg_cfg = cfg.section("telegram")
        self.offset = 0
        self._stop = False
        self.client.register_commands(BOT_COMMANDS)

    # ------------------------------------------------------------ autorisation
    def is_authorized(self, chat_id: str) -> bool:
        """Seul le chat configure a le droit de commander."""
        return bool(self.authorized_chat) and chat_id == self.authorized_chat

    # ---------------------------------------------------------------- reglages
    def paused(self) -> bool:
        return self.pipeline.db.get_meta("notifications_paused") == "1"

    def set_paused(self, value: bool) -> None:
        self.pipeline.db.set_meta("notifications_paused", "1" if value else "0")
        self.sync_control_state()

    def min_edge(self) -> float:
        return float(self.pipeline.cfg.get("app.min_edge", 0.03))

    def set_min_edge(self, value: float) -> None:
        value = float(min(max(value, 0.005), 0.20))
        self.pipeline.db.set_meta("min_edge_override", value)
        self.pipeline.apply_edge_override()
        self.sync_control_state()

    def sync_control_state(self) -> None:
        """Publie l'etat (pause, edge, bot local) dans le depot GitHub."""
        import subprocess
        state = {"paused": self.paused(), "min_edge": self.min_edge(),
                 "local_active": str(self.pipeline.db.get_meta(
                     "notifications_local") or "") == "1"}
        try:
            subprocess.Popen(
                ["python", str(Path(__file__).resolve().parents[1]
                               / "scripts" / "push_control.py"),
                 json.dumps(state, sort_keys=True)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parents[1]))
        except OSError:
            LOGGER.info("push_control indisponible : etat local uniquement")

    def _publish_local(self, active: bool) -> None:
        """Declare qui gere Telegram (bot local ou cycles GitHub).

        Un seul lecteur de messages a la fois : quand le bot local tourne,
        les cycles GitHub n'interrogent plus Telegram (voir cmd_cycle) ;
        quand il s'arrete, GitHub reprend automatiquement.
        """
        self.pipeline.db.set_meta("notifications_local",
                                  "1" if active else "0")
        self.sync_control_state()

    # -------------------------------------------------------------- envois
    def send_digest(self, title: str | None = None) -> int:
        if self.paused():
            LOGGER.info("notifications en pause : digest non envoye")
            return 0
        text = self.pipeline.digest_text(title=title)
        sent = 0
        for chunk in split_message(text):
            self.client.send(chunk, keyboard=build_keyboard())
            sent += 1
            time.sleep(0.6)
        self.pipeline.db.log_event("telegram_digest", f"{sent} message(s)")
        return sent

    def send_stats(self) -> None:
        self.client.send(self.pipeline.stats_text(),
                         keyboard=build_keyboard())

    def send_learning(self) -> None:
        from .learning import learning_summary
        self.client.send(learning_summary(self.pipeline.db),
                         keyboard=build_keyboard())

    # ------------------------------------------------------------ panneau
    def send_control_panel(self, chat_id: str | None = None) -> None:
        """Le panneau de controle avec l'etat courant."""
        paused = self.paused()
        edge = self.min_edge()
        text = (
            "<b>⚙️ Panneau de contrôle</b>\n\n"
            f"· Notifications : <b>{'⏸ EN PAUSE' if paused else '▶️ actives'}</b>\n"
            f"· Seuil d'edge : <b>{edge * 100:.0f}%</b>\n"
            f"· Modèle : <code>{html.escape(self._model_version())}</code>\n\n"
            "Choisissez une action ci-dessous. Tout ce que vous touchez ici "
            "agit immédiatement sur l'agent (local ET GitHub Actions).")
        self.client.send(text, chat_id, keyboard=self.panel_keyboard())

    def panel_keyboard(self) -> list[list[dict]]:
        paused = self.paused()
        pause_btn = ({"text": "▶️ Reprendre les envois", "callback_data": "resume"}
                     if paused else
                     {"text": "⏸ Pause des notifications", "callback_data": "pause"})
        return [
            [{"text": "📜 Digest maintenant", "callback_data": "digest_now"}],
            [{"text": "🔄 Rafraîchir les cotes", "callback_data": "refresh"},
             {"text": "🏋️ Entraîner maintenant", "callback_data": "train_now"}],
            [pause_btn, {"text": "🎯 Seuil d'edge", "callback_data": "edge_menu"}],
            [{"text": "📊 Stats", "callback_data": "stats"},
             {"text": "🧠 Apprentissage", "callback_data": "learning"}],
            [{"text": "🏠 Fermer", "callback_data": "noop"}],
        ]

    def edge_keyboard(self) -> list[list[dict]]:
        current = self.min_edge()
        row: list[dict] = []
        for preset in EDGE_PRESETS:
            mark = "✅ " if abs(preset - current) < 1e-9 else ""
            row.append({"text": f"{mark}{preset * 100:.0f}%",
                        "callback_data": f"edge:{preset:.2f}"})
        return [row, [{"text": "← Retour au panneau", "callback_data": "menu"}]]

    def _model_version(self) -> str:
        model = self.pipeline.db.active_model()
        return model["version"] if model else "aucun"

    # ------------------------------------------------------------ commandes
    def handle_command(self, message: dict) -> None:
        text = (message.get("text") or "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        if not text.startswith("/"):
            return
        if not self.is_authorized(chat_id):
            LOGGER.warning("commande refusee : chat %s non autorise", chat_id)
            return
        command, _, argument = text.partition(" ")
        command = command.split("@")[0].lower()
        LOGGER.info("commande %s de %s", command, chat_id)

        if command in ("/start", "/help"):
            self.client.send(
                "<b>Prematch AI</b> — agent de value betting\n"
                "· /today — les paris du jour\n"
                "· /stats — performance (ROI, P&L)\n"
                "· /learning — ce que l'IA a appris\n"
                "· /menu — panneau de contrôle par boutons\n"
                "· /pause — couper les notifications\n"
                "· /resume — reprendre les notifications\n"
                "· /edge 0.04 — changer le seuil d'edge\n"
                "· /bankroll 1000 — définir la bankroll\n"
                "· /help — cette aide\n\n"
                "Les mises suivent un Kelly fractionné plafonné : "
                "aucune mise folle, jamais.", chat_id,
                keyboard=self.panel_keyboard())
        elif command == "/today":
            self.send_today(chat_id)
        elif command == "/stats":
            self.client.send(self.pipeline.stats_text(), chat_id,
                             keyboard=self.panel_keyboard())
        elif command == "/learning":
            self.send_learning()
        elif command == "/menu":
            self.send_control_panel(chat_id)
        elif command == "/pause":
            self.set_paused(True)
            self.client.send("⏸ Notifications <b>en pause</b>. L'agent continue "
                             "de collecter et d'apprendre, mais n'envoie plus "
                             "rien. /resume pour reprendre.", chat_id)
        elif command == "/resume":
            self.set_paused(False)
            self.client.send("▶️ Notifications <b>réactivées</b>. Prochain digest "
                             "selon l'horaire planifié (ou bouton 📜).", chat_id)
        elif command == "/edge":
            try:
                value = float(argument.split()[0].replace("%", ""))
                value = value / 100 if value > 1 else value
                if not 0.005 <= value <= 0.20:
                    raise ValueError
            except (IndexError, ValueError):
                self.client.send("Usage : /edge 0.04  (ou /edge 4 pour 4 %)",
                                 chat_id)
                return
            self.set_min_edge(value)
            self.client.send(f"🎯 Seuil d'edge fixé à <b>{value * 100:.1f}%</b> "
                             "(persistant, local et GitHub Actions).", chat_id)
        elif command == "/bankroll":
            try:
                amount = float(argument.split()[0])
                if amount <= 0:
                    raise ValueError
                self.pipeline.set_bankroll(amount)
                self.client.send(f"Bankroll mise à jour : {amount:.2f} €",
                                 chat_id)
            except (IndexError, ValueError):
                self.client.send("Usage : /bankroll 1000", chat_id)
        else:
            self.client.send("Commande inconnue. /menu pour le panneau, "
                             "/help pour l'aide.", chat_id)

    # ------------------------------------------------------------ actions
    def send_today(self, chat_id: str | None = None) -> None:
        try:
            result = self.pipeline.predict(store=False)
            for chunk in split_message(self.pipeline.digest_text(result=result)):
                self.client.send(chunk, chat_id=chat_id,
                                 keyboard=build_keyboard())
        except Exception as exc:
            LOGGER.exception("digest impossible")
            self.client.send(f"⚠️ Erreur : {html.escape(str(exc))}", chat_id)

    def do_refresh(self, chat_id: str) -> None:
        self.client.send("🔄 Récupération des cotes fraîches…", chat_id)
        try:
            result = self.pipeline.refresh_fixtures()
            self.client.send(f"✅ Cotes rafraîchies : "
                             f"<code>{html.escape(str(result))}</code>", chat_id)
        except Exception as exc:
            LOGGER.exception("refresh impossible")
            self.client.send(f"⚠️ Erreur : {html.escape(str(exc))}", chat_id)

    def do_train(self, chat_id: str) -> None:
        self.client.send("🏋️ Entraînement lancé (2 à 5 min selon les données)…",
                         chat_id)
        def job() -> None:
            try:
                result = self.pipeline.train()
                metrics = result.get("metrics", {})
                self.client.send(
                    f"✅ Modèle <code>{html.escape(result['version'])}</code> "
                    f"entraîné sur {result['n_train']} matchs · "
                    f"logloss holdout "
                    f"<b>{metrics.get('holdout_logloss_sharp', metrics.get('holdout_logloss', '?'))}</b> "
                    f"· T={metrics.get('temperature', '?')}", chat_id,
                    keyboard=self.panel_keyboard())
            except Exception as exc:
                LOGGER.exception("train impossible")
                self.client.send(f"⚠️ Erreur : {html.escape(str(exc))}", chat_id)
        threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------- polling
    def poll_once(self) -> None:
        data = self.client.call("getUpdates", {
            "offset": self.offset + 1,
            "timeout": 25,
            "allowed_updates": ["message", "callback_query"],
        })
        for update in data:
            self.offset = int(update.get("update_id", self.offset))
            if "message" in update:
                self.handle_command(update["message"])
            elif "callback_query" in update:
                self.handle_callback(update["callback_query"])

    def handle_callback(self, query: dict) -> None:
        action = query.get("data", "")
        chat_id = str(query.get("message", {}).get("chat", {})
                      .get("id", ""))
        if not self.is_authorized(chat_id):
            LOGGER.warning("callback refuse : chat %s non autorise", chat_id)
            try:
                self.client.call("answerCallbackQuery", {
                    "callback_query_id": query["id"],
                    "text": "Non autorisé — configurez TELEGRAM_CHAT_ID."})
            except RuntimeError:
                pass
            return
        try:
            self.dispatch_callback(action, chat_id)
        except Exception as exc:
            LOGGER.exception("callback %s : erreur", action)
            try:
                self.client.send(f"⚠️ Erreur : {html.escape(str(exc))}",
                                 chat_id)
            except RuntimeError:
                pass
        finally:
            try:
                self.client.call("answerCallbackQuery", {
                    "callback_query_id": query["id"]})
            except RuntimeError:
                pass

    def dispatch_callback(self, action: str, chat_id: str) -> None:
        if action == "stats":
            self.send_stats()
        elif action == "today":
            self.send_today(chat_id)
        elif action == "learning":
            self.send_learning()
        elif action == "menu":
            self.send_control_panel(chat_id)
        elif action == "digest_now":
            sent = self.send_digest(title="à la demande")
            self.client.send(
                "📜 Digest envoyé." if sent else
                "⏸ Notifications en pause : réactivez d'abord (bouton du "
                "panneau ou /resume).", chat_id)
        elif action == "refresh":
            self.do_refresh(chat_id)
        elif action == "train_now":
            self.do_train(chat_id)
        elif action == "pause":
            self.set_paused(True)
            self.client.send("⏸ Notifications <b>en pause</b>.", chat_id)
            self.send_control_panel(chat_id)
        elif action == "resume":
            self.set_paused(False)
            self.client.send("▶️ Notifications <b>réactivées</b>.", chat_id)
            self.send_control_panel(chat_id)
        elif action.startswith("edge:"):
            try:
                value = float(action.split(":", 1)[1])
            except ValueError:
                value = self.min_edge()
            self.set_min_edge(value)
            self.client.send(f"🎯 Seuil d'edge fixé à <b>{value * 100:.0f}%</b>. "
                             "S'applique dès le prochain cycle (local + "
                             "GitHub Actions).", chat_id)
            self.client.send("🎯 Choisissez un nouveau seuil ou revenez au "
                             "panneau :", chat_id, keyboard=self.edge_keyboard())
        elif action == "edge_menu":
            self.client.send(f"🎯 Seuil actuel : <b>{self.min_edge() * 100:.0f}%</b> "
                             "(edge minimal pour retenir un pari).", chat_id,
                             keyboard=self.edge_keyboard())
        elif action == "noop":
            return
        else:
            LOGGER.warning("callback inconnu : %s", action)

    def alert_new_value(self, min_edge_alert: float | None = None) -> None:
        """Alerte immedidate si un pick depasse un seuil renforce."""
        if self.paused():
            return
        if not self.tg_cfg.get("alert_new_signals", True):
            return
        threshold = float(min_edge_alert
                          or (self.cfg.get("app.min_edge", 0.03) * 2.0))
        result = self.pipeline.predict(store=True)
        hot = [p for p in result["picks"] if p.edge >= threshold]
        if not hot:
            return
        lines = ["<b>🔥 Value forte détectée</b>"]
        for pick in hot[:3]:
            lines.append(f"· {html.escape(pick.home)} – {html.escape(pick.away)}"
                         f" : {html.escape(pick.label)} @ {pick.price:.2f} "
                         f"(+{pick.edge * 100:.1f}%)")
        self.client.send("\n".join(lines), keyboard=self.panel_keyboard())

    def run_forever(self, check_interval_s: int = 30) -> None:
        """Boucle principale du bot (polling + digests planifies)."""
        digest_hours = [int(h) for h in self.tg_cfg.get("digest_hours", [9, 17])]
        quiet = [int(x) for x in self.cfg.section("scheduler").get(
            "quiet_hours", [1, 7])]
        sent_on: set[str] = set()
        last_refresh = 0.0
        refresh_minutes = int(self.cfg.get("scheduler.refresh_minutes", 30))
        alert_check_minutes = int(
            self.cfg.get("scheduler.alert_check_minutes", 45))
        last_alert = 0.0

        LOGGER.info("bot demarre (digests a %s h)", digest_hours)
        self._publish_local(True)
        try:
            while not self._stop:
                now = time.time()
                local_hour = time.localtime().tm_hour
                try:
                    # rafraichissement des donnees
                    if now - last_refresh > refresh_minutes * 60:
                        self.pipeline.refresh_fixtures()
                        last_refresh = now

                    # digests (respectent la pause et les heures calmes)
                    stamp = time.strftime("%Y-%m-%d-%H")
                    hour = int(stamp.split("-")[-1])
                    if hour in digest_hours and stamp not in sent_on \
                            and not (quiet[0] <= local_hour < quiet[1]):
                        self.send_digest()
                        sent_on.add(stamp)

                    # alertes de value (respectent la pause)
                    if now - last_alert > alert_check_minutes * 60:
                        if not (quiet[0] <= local_hour < quiet[1]):
                            self.alert_new_value()
                        last_alert = now

                    # commandes + boutons
                    self.poll_once()
                except RuntimeError as exc:
                    LOGGER.warning("telegram : %s", exc)
                    time.sleep(15)
                except Exception:
                    LOGGER.exception("boucle bot : erreur inattendue")
                    time.sleep(30)
                time.sleep(check_interval_s)
        finally:
            # l'arret du bot local rend Telegram aux cycles GitHub
            self._publish_local(False)

    def stop(self) -> None:
        self._stop = True
