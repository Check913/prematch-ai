# -*- coding: utf-8 -*-
"""Bot Telegram : digest quotidien, alertes de value et commandes.

Fonctions :
  * digest aux heures configurees (paris du jour, mise Kelly, alternatives) ;
  * alerte immediate quand un nouveau signal depasse un seuil renforce
    (edge >= 2x le seuil de base) ;
  * commandes : /start, /today, /stats, /bankroll N, /help ;
  * boutons inline pour /today : split des picks en messages courts.

Le token et le chat id viennent de l'environnement (jamais en clair).
"""
from __future__ import annotations

import html
import time
from typing import Any

import requests

from .config import Config
from .logging_setup import get_logger
from .pipeline import Pipeline

LOGGER = get_logger("prematch.telegram")

API = "https://api.telegram.org/bot{token}/{method}"


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
             {"text": "🔄 Aujourd'hui", "callback_data": "today"}]]


class TelegramBot:
    """Boucle longue : polling + envoi planifie des digests."""

    def __init__(self, pipeline: Pipeline, cfg: Config):
        self.pipeline = pipeline
        self.cfg = cfg
        token = cfg.telegram_token()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN absent de l'environnement")
        self.client = TelegramClient(token, cfg.telegram_chat_id())
        self.tg_cfg = cfg.section("telegram")
        self.offset = 0
        self._stop = False

    # -------------------------------------------------------------- envois
    def send_digest(self, title: str | None = None) -> int:
        text = self.pipeline.digest_text(title=title)
        sent = 0
        for chunk in split_message(text):
            self.client.send(chunk, keyboard=build_keyboard())
            sent += 1
            time.sleep(0.6)
        self.pipeline.db.log_event("telegram_digest", f"{sent} message(s)")
        return sent

    def send_stats(self) -> None:
        self.client.send(self.pipeline.stats_text())

    def send_learning(self) -> None:
        from .learning import learning_summary
        self.client.send(learning_summary(self.pipeline.db))

    # ------------------------------------------------------------ commandes
    def handle_command(self, message: dict) -> None:
        text = (message.get("text") or "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        if not text.startswith("/"):
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
                "· /bankroll 1000 — definir la bankroll\n"
                "· /help — cette aide\n\n"
                "Le digest est envoye automatiquement chaque jour. "
                "Les mises suivent un Kelly fractionne plafonné : "
                "aucune mise folle, jamais.", chat_id)
        elif command == "/today":
            try:
                result = self.pipeline.predict(store=False)
                for chunk in split_message(
                        self.pipeline.digest_text(result=result)):
                    self.client.send(chunk, chat_id=chat_id,
                                     keyboard=build_keyboard())
            except Exception as exc:
                LOGGER.exception("digest impossible")
                self.client.send(f"⚠️ Erreur : {html.escape(str(exc))}",
                                 chat_id)
        elif command == "/stats":
            self.client.send(self.pipeline.stats_text(), chat_id)
        elif command == "/learning":
            self.send_learning()
        elif command == "/bankroll":
            try:
                amount = float(argument.split()[0])
                if amount <= 0:
                    raise ValueError
                self.pipeline.set_bankroll(amount)
                self.client.send(f"Bankroll mise a jour : {amount:.2f} €",
                                 chat_id)
            except (IndexError, ValueError):
                self.client.send("Usage : /bankroll 1000", chat_id)
        else:
            self.client.send("Commande inconnue. /help pour l'aide.", chat_id)

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
                query = update["callback_query"]
                action = query.get("data", "")
                chat_id = str(query.get("message", {}).get("chat", {})
                              .get("id", ""))
                if action == "stats":
                    self.client.send(self.pipeline.stats_text(), chat_id)
                elif action == "today":
                    result = self.pipeline.predict(store=False)
                    for chunk in split_message(
                            self.pipeline.digest_text(result=result)):
                        self.client.send(chunk, chat_id=chat_id,
                                         keyboard=build_keyboard())
                try:
                    self.client.call("answerCallbackQuery", {
                        "callback_query_id": query["id"]})
                except RuntimeError:
                    pass

    def alert_new_value(self, min_edge_alert: float | None = None) -> None:
        """Alerte immedidate si un pick depasse un seuil renforce."""
        if not self.tg_cfg.get("alert_new_signals", True):
            return
        threshold = float(min_edge_alert
                          or (self.cfg.get("app.min_edge", 0.03) * 2.0))
        result = self.pipeline.predict(store=True)
        hot = [p for p in result["picks"] if p.edge >= threshold]
        if not hot:
            return
        lines = ["<b>🔥 Value forte detectee</b>"]
        for pick in hot[:3]:
            lines.append(f"· {html.escape(pick.home)} – {html.escape(pick.away)}"
                         f" : {html.escape(pick.label)} @ {pick.price:.2f} "
                         f"(+{pick.edge * 100:.1f}%)")
        self.client.send("\n".join(lines))

    def run_forever(self, check_interval_s: int = 30) -> None:
        """Boucle principale du bot (polling + digests planifies)."""
        import threading

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
        while not self._stop:
            now = time.time()
            local_hour = time.localtime().tm_hour
            try:
                # rafraichissement des donnees
                if now - last_refresh > refresh_minutes * 60:
                    self.pipeline.refresh_fixtures()
                    last_refresh = now

                # digests
                stamp = time.strftime("%Y-%m-%d-%H")
                hour = int(stamp.split("-")[-1])
                if hour in digest_hours and stamp not in sent_on \
                        and not (quiet[0] <= local_hour < quiet[1]):
                    self.send_digest()
                    sent_on.add(stamp)

                # alertes de value
                if now - last_alert > alert_check_minutes * 60:
                    if not (quiet[0] <= local_hour < quiet[1]):
                        self.alert_new_value()
                    last_alert = now

                # commandes
                self.poll_once()
            except RuntimeError as exc:
                LOGGER.warning("telegram : %s", exc)
                time.sleep(15)
            except Exception:
                LOGGER.exception("boucle bot : erreur inattendue")
                time.sleep(30)
            time.sleep(check_interval_s)

    def stop(self) -> None:
        self._stop = True
