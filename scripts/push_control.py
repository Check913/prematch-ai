# -*- coding: utf-8 -*-
"""Publie l'etat de controle (pause / seuil d'edge) dans le depot GitHub.

Le bot Telegram local appelle ce script apres chaque changement fait par
les boutons ; le workflow GitHub Actions lit control.json a chaque cycle,
pour que les reglages suivent partout (PC eteint ou non).

Best effort : echoue silencieusement si `gh` est absent ou hors ligne —
l'etat reste alors local (la base) et se re-synchronisera plus tard.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "control.json"
POLL = 30  # s


def repo_slug() -> str | None:
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"],
                             cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             timeout=POLL).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if "github.com" not in url:
        return None
    path = url.split("github.com", 1)[-1]
    slug = path.strip("/: ").removesuffix(".git").strip("/")
    return slug if "/" in slug else None


def publish(state: dict) -> bool:
    slug = repo_slug()
    if not slug:
        return False
    payload = dict(state)
    payload["at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds")
    CONTROL.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    content = base64.b64encode(
        CONTROL.read_bytes()).decode("ascii")
    sha = None
    probe = subprocess.run(
        ["gh", "api", f"repos/{slug}/contents/control.json", "--jq", ".sha"],
        capture_output=True, text=True, timeout=POLL, stdin=subprocess.DEVNULL)
    if probe.returncode == 0:
        sha = probe.stdout.strip()
    body: dict[str, str] = {
        "message": f"control: {json.dumps(payload, sort_keys=True)[:120]}",
        "content": content, "branch": "main"}
    if sha:
        body["sha"] = sha
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        tmp = f.name
    push = subprocess.run(
        ["gh", "api", f"repos/{slug}/contents/control.json",
         "-X", "PUT", "--input", tmp],
        capture_output=True, text=True, timeout=2 * POLL,
        stdin=subprocess.DEVNULL)
    Path(tmp).unlink(missing_ok=True)
    if push.returncode != 0:
        print(f"push control echoue : {push.stderr.strip()[:200]}")
        return False
    return True


if __name__ == "__main__":
    import sys
    state = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    ok = publish(state)
    print("control.json publie" if ok else "control.json : publication ratee")
    raise SystemExit(0 if ok else 1)
