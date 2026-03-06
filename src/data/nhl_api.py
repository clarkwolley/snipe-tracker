"""
NHL API client.

Talks to the free NHL API (api-web.nhle.com) to fetch schedules,
standings, team rosters, and game boxscores. No API key needed.

💡 KEY CONCEPT: This module is a "data access layer" — its only job
is to fetch raw data from the API and return it as Python dicts.
It doesn't transform, filter, or analyze anything. That separation
keeps the code clean and testable.
"""

import requests
from src.config import NHL_API_BASE


DEFAULT_TIMEOUT = 10  # seconds


class NHLApiError(Exception):
    """Raised when an NHL API request fails."""


MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]  # seconds to wait between retries


def _get(endpoint: str) -> dict:
    """
    Make a GET request to the NHL API with automatic retry on 429.

    Args:
        endpoint: API path (e.g., '/schedule/now')

    Returns:
        Parsed JSON response as a dict.

    Raises:
        NHLApiError: If the request fails after all retries.
    """
    import time

    url = f"{NHL_API_BASE}{endpoint}"

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                print(f"  ⏳ Rate limited, retrying in {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise NHLApiError(f"NHL API request failed: {url} — {e}") from e
        except requests.RequestException as e:
            raise NHLApiError(f"NHL API request failed: {url} — {e}") from e


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
    return _get(f"/schedule/{date}")


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
    return _get("/standings/now")


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
    return _get(f"/club-stats/{team_abbrev}/now")


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
    return _get(f"/gamecenter/{game_id}/boxscore")


def get_team_schedule(team_abbrev: str, season: str = "20242025") -> dict:
    """
    Fetch a team's full season schedule (past + future games).

    Args:
        team_abbrev: Three-letter team code (e.g., 'COL')
        season: Season string like '20242025'

    Returns:
        Dict with 'games' list — each game has IDs, dates, opponents,
        and scores (for completed games).
    """
    return _get(f"/club-schedule-season/{team_abbrev}/{season}")


def get_player_game_log(player_id: int, season: str = "20242025") -> dict:
    """
    Fetch a player's game-by-game log for a season.

    Works for both skaters and goalies.

    Args:
        player_id: NHL player ID
        season: Season string like '20242025'

    Returns:
        Dict with 'gameLog' list of per-game stat lines.
    """
    return _get(f"/player/{player_id}/game-log/{season}/2")
