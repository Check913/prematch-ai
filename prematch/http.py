"""Client HTTP « poli » : cache disque, delai par domaine, robots.txt, budget.

Objectif : rendre le scraping continu supportable pour les sources et
previsible pour l'operateur.

Regles appliquees :
  * un delai minimum + jitter entre deux requetes vers un meme hote ;
  * un budget horaire par hote (au-dela : RateLimitExceeded, on recule) ;
  * respect de robots.txt (si indisponible : on autorise, mais on log) ;
  * re-essais avec backoff exponentiel sur 429/5xx et erreurs reseau ;
  * cache disque avec TTL pour ne pas re-telecharger les memes pages.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from .logging_setup import get_logger

LOGGER = get_logger("prematch.http")

USER_AGENT = ("PrematchAI/1.0 (agent d'analyse sportive personnel; "
              "+contact via Telegram)")
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """Echec d'acces HTTP definitif."""


class RobotsDisallowed(FetchError):
    """robots.txt interdit explicitement l'URL demandee."""


class RateLimitExceeded(FetchError):
    """Budget de requetes par hote depasse : il faut attendre."""


class PoliteClient:
    """Client HTTP partage par toutes les sources."""

    def __init__(self, cache_dir: Path | None = None, timeout: int = 30,
                 min_delay: float = 2.0, jitter: float = 1.0,
                 max_retries: int = 3, respect_robots: bool = True,
                 cache_ttl: int = 900,
                 max_requests_per_hour: int = 240,
                 user_agent: str = USER_AGENT,
                 session: requests.Session | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_delay = max(0.0, float(min_delay))
        self.jitter = max(0.0, float(jitter))
        self.max_retries = max(1, int(max_retries))
        self.respect_robots = respect_robots
        self.cache_ttl = int(cache_ttl)
        self.max_per_hour = int(max_requests_per_hour)
        self.user_agent = user_agent
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/csv,application/json,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        })
        self._last_request: dict[str, float] = {}
        self._window: dict[str, deque[float]] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self.nb_requests = 0
        self.nb_cache_hits = 0

    # ------------------------------------------------------------------ cache
    def _cache_file(self, url: str, suffix: str) -> Path:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}{suffix}"

    def _cache_read(self, url: str, ttl: int) -> bytes | None:
        if self.cache_dir is None or ttl <= 0:
            return None
        body_path = self._cache_file(url, ".bin")
        meta_path = self._cache_file(url, ".meta.json")
        if not body_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if meta.get("url") != url:
            return None
        age = time.time() - float(meta.get("fetched_at", 0))
        if age > ttl:
            return None
        try:
            payload = body_path.read_bytes()
        except OSError:
            return None
        self.nb_cache_hits += 1
        LOGGER.debug("cache hit (%.0fs) %s", age, url)
        return payload

    def _cache_write(self, url: str, payload: bytes,
                     content_type: str = "") -> None:
        if self.cache_dir is None:
            return
        try:
            self._cache_file(url, ".bin").write_bytes(payload)
            self._cache_file(url, ".meta.json").write_text(json.dumps({
                "url": url,
                "fetched_at": time.time(),
                "content_type": content_type,
                "size": len(payload),
            }), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disque plein / lock
            LOGGER.warning("ecriture cache impossible (%s) : %s", url, exc)

    # ----------------------------------------------------------------- robots
    def _robots_for(self, host: str, scheme: str) -> RobotFileParser | None:
        if host in self._robots:
            return self._robots[host]
        parser: RobotFileParser | None = None
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            resp = self.session.get(robots_url, timeout=min(self.timeout, 15))
            if resp.status_code == 200:
                parser = RobotFileParser()
                parser.parse(resp.text.splitlines())
            else:
                LOGGER.debug("robots.txt %s -> HTTP %s (autorise)",
                             host, resp.status_code)
        except requests.RequestException as exc:
            LOGGER.debug("robots.txt %s illisible (%s) -> autorise", host, exc)
        self._robots[host] = parser
        return parser

    def _check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return
        parsed = urlparse(url)
        parser = self._robots_for(parsed.netloc, parsed.scheme or "https")
        if parser is not None and not parser.can_fetch(self.user_agent, url):
            raise RobotsDisallowed(f"robots.txt interdit {url}")

    # ------------------------------------------------------------- throttling
    def _throttle(self, host: str) -> None:
        now = time.time()
        window = self._window.setdefault(host, deque())
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= self.max_per_hour:
            retry_in = 3600 - (now - window[0]) + 1
            raise RateLimitExceeded(
                f"budget horaire atteint pour {host} "
                f"(reprendre dans {retry_in:.0f}s)")
        last = self._last_request.get(host, 0.0)
        wait = self.min_delay - (now - last)
        if wait > 0:
            time.sleep(wait + random.uniform(0, self.jitter))
        self._last_request[host] = time.time()
        window.append(time.time())

    # ------------------------------------------------------------------- core
    def get_bytes(self, url: str, ttl: int | None = None, force: bool = False,
                  headers: dict[str, str] | None = None) -> bytes:
        """Telecharge une URL (ou sort du cache) et renvoie le corps brut."""
        effective_ttl = self.cache_ttl if ttl is None else int(ttl)
        if not force and effective_ttl > 0:
            cached = self._cache_read(url, effective_ttl)
            if cached is not None:
                return cached

        self._check_robots(url)
        host = urlparse(url).netloc.lower()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle(host)
            try:
                resp = self.session.get(url, timeout=self.timeout,
                                        headers=headers)
                self.nb_requests += 1
            except requests.RequestException as exc:
                last_error = exc
                LOGGER.warning("reseau (%s/%s) %s : %s",
                               attempt, self.max_retries, url, exc)
                time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 1.5))
                continue

            if resp.status_code == 200:
                payload = resp.content
                self._cache_write(url, payload,
                                  resp.headers.get("Content-Type", ""))
                return payload
            if resp.status_code in RETRY_STATUSES:
                last_error = FetchError(f"HTTP {resp.status_code} pour {url}")
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if (retry_after or "").isdigit() \
                    else min(60.0, 3.0 * attempt) + random.uniform(0, 2.0)
                LOGGER.warning("HTTP %s (%s/%s) %s -> pause %.0fs",
                               resp.status_code, attempt, self.max_retries,
                               url, delay)
                time.sleep(delay)
                continue
            raise FetchError(f"HTTP {resp.status_code} pour {url}")

        raise FetchError(f"abandon apres {self.max_retries} tentatives : "
                         f"{url} ({last_error})")

    def get_text(self, url: str, ttl: int | None = None, force: bool = False,
                 encoding: str | None = None) -> str:
        payload = self.get_bytes(url, ttl=ttl, force=force)
        if encoding:
            return payload.decode(encoding, errors="replace")
        for candidate in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return payload.decode(candidate)
            except UnicodeDecodeError:
                continue
        return payload.decode("utf-8", errors="replace")

    def get_json(self, url: str, ttl: int | None = None,
                 force: bool = False) -> object:
        text = self.get_text(url, ttl=ttl, force=force, encoding="utf-8")
        try:
            return json.loads(text)
        except ValueError as exc:
            raise FetchError(f"JSON invalide pour {url} : {exc}") from exc

    def stats(self) -> dict[str, int]:
        return {"requests": self.nb_requests, "cache_hits": self.nb_cache_hits}
