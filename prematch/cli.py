# -*- coding: utf-8 -*-
"""Interface en ligne de commande de Prematch AI.

Exemples :
  python -m prematch demo                      # donnees synthetiques + train
  python -m prematch ingest-history            # telecharge l'historique reel
  python -m prematch ingest-fixtures           # matchs a venir + cotes
  python -m prematch train                     # entraine le modele
  python -m prematch backtest --edge 0.04      # backtest walk-forward
  python -m prematch today                     # paris du jour (console)
  python -m prematch show --match KEY          # analyse detaillee d'un match
  python -m prematch status                    # sante du systeme
  python -m prematch bot                       # bot Telegram 24/7
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .logging_setup import setup_logging


def _pipeline(args):
    cfg = load_config(getattr(args, "config", None))
    setup_logging(cfg.get("app.log_level", "INFO"))
    from .pipeline import Pipeline
    return cfg, Pipeline(cfg)


def _print_pick_line(index: int, pick) -> str:
    return (f"{index:2d}. [{pick.kickoff[:16]}] {pick.home} - {pick.away} "
            f"({pick.div}) | {pick.label} @ {pick.price:.2f} ({pick.book}) | "
            f"edge {pick.edge * 100:+.1f}% | mise {pick.stake:.2f}")


def cmd_demo(args) -> int:
    cfg, pipeline = _pipeline(args)
    cfg.data["sources"]["demo"]["enabled"] = True
    cfg.data["sources"]["football_data"]["enabled"] = False
    counts = pipeline.ingest_history()
    print(f"demo : {counts}")
    trained = pipeline.train()
    print(f"modele : {trained['version']} "
          f"(logloss {trained['metrics'].get('holdout_logloss', 'n/a')})")
    result = pipeline.predict(days_ahead=10)
    print(_format_digest_plain(pipeline, result))
    return 0


def cmd_ingest_history(args) -> int:
    _, pipeline = _pipeline(args)
    result = pipeline.ingest_history()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 1


def cmd_ingest_fixtures(args) -> int:
    _, pipeline = _pipeline(args)
    result = pipeline.refresh_fixtures()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    _, pipeline = _pipeline(args)
    result = pipeline.train()
    print(f"modele {result['version']} entraine sur {result['n_train']} matchs")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return 0


def cmd_backtest(args) -> int:
    cfg, pipeline = _pipeline(args)
    from .backtest import edge_sweep, run_walk_forward
    if args.sweep:
        rows = edge_sweep(pipeline.db, cfg)
        for row in rows:
            print(row)
        return 0
    result = run_walk_forward(pipeline.db, cfg, min_edge=args.edge)
    print(result.summary())
    if args.save:
        edge_label = f"{args.edge:.2f}" if args.edge is not None else "defaut"
        out = pipeline.cfg.reports_dir / f"backtest_{edge_label}.csv"
        result.bet_log.to_csv(out, index=False)
        equity_out = pipeline.cfg.reports_dir / f"equity_{edge_label}.csv"
        result.equity.to_csv(equity_out, index=False, header=["bankroll"])
        print(f"detail : {out}")
        print(f"courbe de bankroll : {equity_out}")
    return 0


def cmd_today(args) -> int:
    _, pipeline = _pipeline(args)
    result = pipeline.predict(days_ahead=args.days)
    print(_format_digest_plain(pipeline, result))
    return 0


def cmd_show(args) -> int:
    _, pipeline = _pipeline(args)
    rows = pipeline.upcoming_features(days_ahead=args.days)
    target = rows[rows["key"].str.contains(args.match, case=False, na=False)]
    if target.empty:
        print(f"aucun match a venir ne correspond a '{args.match}'")
        return 1
    bundle = pipeline.load_bundle()
    results = bundle.market_probabilities(target)
    kickoffs = pipeline.kickoff_map(args.days)
    for item in results:
        item["kickoff"] = kickoffs.get(item["key"], item["date"])
        odds_rows = pipeline.db.latest_odds(item["key"])
        analysis = __import__("prematch.value", fromlist=["value"]) \
            .analyse_match(item, odds_rows)
        print(f"\n=== {item['home']} - {item['away']} "
              f"({item['div']}) {item['kickoff']} ===")
        p = item["p_1x2"]
        print(f"1N2 modele : {p[0] * 100:.1f}% / {p[1] * 100:.1f}% / "
              f"{p[2] * 100:.1f}%  (accord modeles "
              f"{analysis['agreement'] * 100:.0f}%)")
        print(f"buts attendus : {item['lam_home']:.2f} - "
              f"{item['lam_away']:.2f}")
        if not analysis["edges"]:
            print("aucune cote connue pour ce match")
            continue
        print(f"{'selection':34s} {'cote':>6s} {'p.modele':>9s} "
              f"{'p.marche':>9s} {'edge':>8s}")
        for edge in analysis["edges"][:12]:
            label = edge_label(edge, item)
            print(f"{label:34s} {edge['price']:6.2f} "
                  f"{edge['p_model'] * 100:8.1f}% "
                  f"{edge['p_market'] * 100:8.1f}% {edge['edge'] * 100:+7.1f}%")
    return 0


def edge_label(edge: dict, item: dict) -> str:
    from .markets import selection_label
    return selection_label(edge["market"], edge["selection"], edge["line"],
                           item["home"], item["away"])


def cmd_status(args) -> int:
    _, pipeline = _pipeline(args)
    db = pipeline.db
    print(f"matchs : {db.count_matches()}")
    upcoming = len(db.upcoming_matches(since=__import__('datetime').date.today().isoformat()))
    print(f"matchs a venir : {upcoming}")
    model = db.active_model()
    if model:
        print(f"modele actif : {model['version']} ({model['created_at']})")
    else:
        print("modele actif : AUCUN (lancez `prematch train`)")
    stats = db.bet_stats()
    print(f"paris regles : {stats['n']} | ROI {stats['roi']:+.2f}% | "
          f"P&L {stats['pnl']:+.2f} EUR")
    print(f"bankroll : {pipeline.bankroll():.2f} EUR")
    for event in db.recent_events(limit=5):
        print(f"[{event['ts']}] {event['kind']}: {event['message'][:90]}")
    return 0


def cmd_bot(args) -> int:
    _, pipeline = _pipeline(args)
    from .telegram_bot import TelegramBot
    bot = TelegramBot(pipeline, load_config(args.config))
    bot.run_forever()
    return 0


def cmd_cycle(args) -> int:
    """Un cycle unique, pense pour GitHub Actions (toutes les 25 min) :
    reglement des paris, apprentissage si du, reponse aux commandes et
    boutons, digest si l'heure est due, alertes de value forte.
    Si le bot local est actif (control.json), le telegram est desactive
    ici pour eviter que deux lecteurs se disputent les messages.
    """
    cfg, pipeline = _pipeline(args)
    from .learning import daily_cycle
    outcome: dict[str, Any] = {}
    try:
        outcome = daily_cycle(pipeline)
    except Exception as exc:
        print(f"cycle apprentissage : {exc}")
    sent = "rien"
    telegram_skipped = str(pipeline.db.get_meta("notifications_local") or "") == "1"
    if telegram_skipped:
        print("telegram : gere par le bot local (control.json)")
        print(json.dumps({k: outcome.get(k) for k in ("settled", "retrained", "promoted")},
                         default=str))
        return 0
    try:
        from .telegram_bot import TelegramBot
        bot = TelegramBot(pipeline, load_config(args.config))
        quiet = [int(x) for x in cfg.section("scheduler").get(
            "quiet_hours", [1, 7])]
        hour = __import__("time").localtime().tm_hour
        in_quiet = quiet[0] <= hour < quiet[1]
        # 1) repondre aux commandes / boutons arrives depuis le cycle precedent
        bot.poll_once()
        # 2) digest si l'heure planifiee est due et pas encore envoyee
        digest_hours = [int(h) for h in cfg.section("telegram").get(
            "digest_hours", [9, 17])]
        stamp = __import__("time").strftime("%Y-%m-%d-%H")
        last_digest = str(pipeline.db.get_meta("last_digest_hour") or "")
        if int(stamp.split("-")[-1]) in digest_hours and not in_quiet \
                and stamp != last_digest:
            n = bot.send_digest()
            pipeline.db.set_meta("last_digest_hour", stamp)
            sent = f"digest horaire ({n} message(s))"
        # 3) alertes de value forte (edge >= 2x le seuil)
        elif not in_quiet:
            bot.alert_new_value()
            sent = "alertes de value (si edge>=2x le seuil)"
        # 4) preuve de vie sur declenchement manuel
        if str(__import__("os").environ.get("PREMATCH_HEALTH", "")) == "1":
            model = pipeline.db.active_model()
            bot.client.send(
                "🔄 Cycle GitHub OK — modèle "
                f"<code>{model['version'] if model else '?'}</code>, "
                f"edge {bot.min_edge() * 100:.0f}%, "
                f"notifications "
                f"{'en pause' if bot.paused() else 'actives'}.",
                bot.authorized_chat, keyboard=bot.panel_keyboard())
    except Exception as exc:
        print(f"telegram : {exc}")
    print(f"envoi : {sent}")
    print(json.dumps({k: outcome.get(k) for k in ("settled", "retrained", "promoted")},
                     default=str))
    return 0


def cmd_syncctl(args) -> int:
    """Applique control.json (publie par les boutons Telegram) a la base."""
    _, pipeline = _pipeline(args)
    pipeline.apply_control_file()
    print("control.json applique : "
          f"pause={pipeline.db.get_meta('notifications_paused') == '1'} "
          f"| min_edge={pipeline.min_edge_override() or 'config'}")
    return 0


def cmd_digest(args) -> int:
    _, pipeline = _pipeline(args)
    result = pipeline.predict(days_ahead=args.days)
    print(_format_digest_plain(pipeline, result))
    return 0


def _format_digest_plain(pipeline, result) -> str:
    lines = [f"=== PREMATCH AI — {result['model_version']} ===",
             f"{result['n_matches']} matchs analyses"]
    if not result["picks"]:
        lines.append("Aucun pick ne depasse les filtres de valeur.")
        return "\n".join(lines)
    for index, pick in enumerate(result["picks"], start=1):
        lines.append(_print_pick_line(index, pick))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prematch",
                                     description="Agent IA de value betting")
    parser.add_argument("--config", "-c", help="chemin config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="pipeline complet sur donnees synthetiques")
    p = sub.add_parser("ingest-history", help="historique + cotes")
    sub.add_parser("ingest-fixtures", help="matchs a venir + cotes")
    sub.add_parser("train", help="entraine et enregistre un modele")
    p = sub.add_parser("backtest", help="backtest walk-forward")
    p.add_argument("--edge", type=float, default=None,
                   help="seuil d'edge (defaut : config)")
    p.add_argument("--sweep", action="store_true",
                   help="teste plusieurs seuils")
    p.add_argument("--save", action="store_true",
                   help="sauvegarde le detail des paris en CSV")
    p = sub.add_parser("today", help="paris du jour")
    p.add_argument("--days", type=int, default=3)
    p = sub.add_parser("digest", help="texte du digest (console)")
    p.add_argument("--days", type=int, default=3)
    p = sub.add_parser("show", help="analyse detaillee d'un match")
    p.add_argument("match", help="fragment de nom d'equipe ou de cle")
    p.add_argument("--days", type=int, default=3)
    sub.add_parser("status", help="sante du systeme")
    sub.add_parser("bot", help="demarre le bot Telegram")
    sub.add_parser("cycle", help="un cycle unique (reglement + apprentissage "
                                   "+ alertes) - pense pour cron/GitHub Actions")
    sub.add_parser("syncctl", help="applique control.json (reglages Telegram "
                                   "publies) dans la base locale")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "demo": cmd_demo, "ingest-history": cmd_ingest_history,
        "ingest-fixtures": cmd_ingest_fixtures, "train": cmd_train,
        "backtest": cmd_backtest, "today": cmd_today, "show": cmd_show,
        "status": cmd_status, "bot": cmd_bot, "digest": cmd_digest,
        "cycle": cmd_cycle, "syncctl": cmd_syncctl,
    }
    handler = commands[args.command]
    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
