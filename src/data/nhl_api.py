"""
NHL API client.

Talks to the free NHL API (api-web.nhle.com) to fetch schedules,
standings, team rosters, and game boxscores. No API key needed.

💡 KEY CONCEPT: This module is a "data access layer" — its only job
is to fetch raw data from the API and return it as Python dicts.
It doesn't transform, filter, or analyze anything. That separation
keeps the code clean and testable.

Rate limiting strategy:
- Proactive: minimum delay between requests (don't anger the API)
- Reactive: exponential backoff + jitter on 429 responses
- Caching: TTL-based in-memory cache for repeated endpoints
"""

import random
import time
import threading
from typing import Any

import requests
from src.config import NHL_API_BASE, NHL_STATS_BASE, CURRENT_SEASON


DEFAULT_TIMEOUT = 10  # seconds


class NHLApiError(Exception):
    """Raised when an NHL API request fails."""


# --- Rate limiting -----------------------------------------------------------

MAX_RETRIES = 5
BASE_BACKOFF = 10       # initial wait on 429 (seconds)
MAX_BACKOFF = 120       # cap retry wait at 2 minutes
MIN_REQUEST_GAP = 0.35  # minimum seconds between ANY two requests

_last_request_time = 0.0
_rate_lock = threading.Lock()


def _throttle() -> None:
    """Enforce minimum gap between requests (proactive rate limiting)."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < MIN_REQUEST_GAP:
            time.sleep(MIN_REQUEST_GAP - elapsed)
        _last_request_time = time.monotonic()


# --- Response cache ----------------------------------------------------------

DEFAULT_CACHE_TTL = 300  # 5 minutes

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: str) -> dict | None:
    """Return cached response if still fresh, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.monotonic() - entry["ts"] < entry["ttl"]:
            return entry["data"]
        _cache.pop(key, None)
        return None


def _cache_set(key: str, data: dict, ttl: float = DEFAULT_CACHE_TTL) -> None:
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.monotonic(), "ttl": ttl}


def clear_cache() -> None:
    """Clear the in-memory response cache."""
    with _cache_lock:
        _cache.clear()


# --- Core request function ---------------------------------------------------


def _get(endpoint: str, cache_ttl: float = DEFAULT_CACHE_TTL) -> dict:
    """
    Make a GET request to the NHL API.

    Features:
    - In-memory cache with configurable TTL
    - Proactive rate limiting (minimum gap between requests)
    - Exponential backoff + jitter on 429 (rate limited)
    - Respects Retry-After header when present

    Args:
        endpoint: API path (e.g., '/schedule/now')
        cache_ttl: How long to cache this response (seconds).
                   Set to 0 to skip caching.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        NHLApiError: If the request fails after all retries.
    """
    # Check cache first
    if cache_ttl > 0:
        cached = _cache_get(endpoint)
        if cached is not None:
            return cached

    url = f"{NHL_API_BASE}{endpoint}"
    return _fetch(url, endpoint, cache_ttl)


def _get_stats(endpoint: str, cache_ttl: float = DEFAULT_CACHE_TTL) -> dict:
    """
    Make a GET request to the NHL *stats* API (api.nhle.com).

    Same caching and retry logic as _get(), just a different base URL.
    The stats API hosts team-level aggregates (PP%, PK%, faceoff%, etc.)
    that aren't available in the main web API.
    """
    cache_key = f"stats:{endpoint}"
    if cache_ttl > 0:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    url = f"{NHL_STATS_BASE}{endpoint}"
    data = _fetch(url, cache_key, cache_ttl)
    return data


def _fetch(url: str, cache_key: str, cache_ttl: float) -> dict:
    """Shared fetch logic with retries, backoff, and caching."""
    for attempt in range(MAX_RETRIES + 1):
        _throttle()
        try:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Cache successful response
            if cache_ttl > 0:
                _cache_set(cache_key, data, cache_ttl)

            return data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < MAX_RETRIES:
                # Respect Retry-After header if the API sends one
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    # Exponential backoff + jitter
                    wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                    wait += random.uniform(0, wait * 0.25)  # up to 25% jitter

                print(
                    f"  ⏳ Rate limited, retrying in {wait:.0f}s... "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue
            raise NHLApiError(f"NHL API request failed: {url} — {e}") from e
        except requests.RequestException as e:
            raise NHLApiError(f"NHL API request failed: {url} — {e}") from e


# --- Public API functions ----------------------------------------------------
# Each function gets a cache_ttl tuned to how often that data changes.


def get_schedule(date: str = "now") -> dict:
    """
    Fetch the game schedule.

    Args:
        date: Date string 'YYYY-MM-DD' or 'now' for today's schedule.

    Returns:
        Schedule dict with 'gameWeek' containing list of days,
        each day has a list of 'games'.

    💡 CONCEPT: The schedule tells us which teams play on a given
    night. This is our starting point for daily predictions —
    we can't predict games that aren't being played!
    """
    return _get(f"/schedule/{date}", cache_ttl=300)


def get_standings() -> dict:
    """
    Fetch current league standings.

    Returns:
        Standings dict with 'standings' list — one entry per team
        with wins, losses, goals for/against, etc.

    💡 CONCEPT: Standings give us team-level strength indicators.
    A team's win%, goal differential, and home/road splits are
    strong predictors for game outcomes.
    """
    # Standings barely change — cache 10 min
    return _get("/standings/now", cache_ttl=600)


def get_team_roster_stats(team_abbrev: str) -> dict:
    """
    Fetch season stats for all players on a team.

    Args:
        team_abbrev: Three-letter team code (e.g., 'COL', 'TOR', 'EDM')

    Returns:
        Dict with 'skaters' and 'goalies' lists, each containing
        season-level stats (goals, assists, shots, etc.)

    💡 CONCEPT: These are "cumulative" stats — totals for the season.
    We'll later convert these into per-game rates and rolling averages
    in the feature engineering phase.
    """
    return _get(f"/club-stats/{team_abbrev}/now", cache_ttl=600)


def get_boxscore(game_id: int) -> dict:
    """
    Fetch the boxscore for a specific game.

    Args:
        game_id: NHL game ID (e.g., 2024020949)

    Returns:
        Full boxscore with team totals and individual player stats
        (goals, shots, TOI, hits, etc.)

    💡 CONCEPT: Boxscores are the richest data source — they tell us
    exactly what every player did in a specific game. This is the
    training data for our goal scorer model: "given these conditions,
    did this player score?"
    """
    # Completed game boxscores never change — cache aggressively
    return _get(f"/gamecenter/{game_id}/boxscore", cache_ttl=3600)


def get_team_schedule(team_abbrev: str, season: str = CURRENT_SEASON) -> dict:
    """
    Fetch a team's full season schedule (past + future games).

    Args:
        team_abbrev: Three-letter team code (e.g., 'COL')
        season: Season string like '20242025'

    Returns:
        Dict with 'games' list — each game has IDs, dates, opponents,
        and scores (for completed games).
    """
    return _get(f"/club-schedule-season/{team_abbrev}/{season}", cache_ttl=600)


def get_player_game_log(player_id: int, season: str = CURRENT_SEASON) -> dict:
    """
    Fetch a player's game-by-game log for a season.

    Works for both skaters and goalies.

    Args:
        player_id: NHL player ID
        season: Season string like '20242025'

    Returns:
        Dict with 'gameLog' list of per-game stat lines.
    """
    return _get(f"/player/{player_id}/game-log/{season}/2", cache_ttl=600)

def get_team_stats_summary(season: str = CURRENT_SEASON) -> dict:
    """
    Fetch team-level stat summaries (PP%, PK%, faceoff%, etc.).

    This data lives on the *stats* API (api.nhle.com) — a different
    endpoint than the main web API. The standings endpoint doesn't
    include special teams data, so we need this to get real PP/PK numbers.

    Args:
        season: Season ID like '20252026'

    Returns:
        Dict with 'data' list — one entry per team with:
        - teamFullName, teamId
        - powerPlayPct, penaltyKillPct
        - goalsForPerGame, goalsAgainstPerGame
        - faceoffWinPct, shotsForPerGame, etc.
    """
    return _get_stats(
        f"/team/summary?cayenneExp=seasonId={season}",
        cache_ttl=600,  # 10 min — same as standings
    )
