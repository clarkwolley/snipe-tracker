"""
Historical data collector — fetches boxscores for past games.

💡 KEY CONCEPT: Machine learning needs LOTS of examples to learn from.
One game is an anecdote. 1,000 games is a dataset. This module walks
through past games and collects player-level stats from each one.

We store results as CSV files so we don't have to re-fetch from the
API every time we want to retrain the model.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

from src.data import nhl_api
from src.data.collector import get_game_player_stats


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _ensure_data_dir():
    """Create the data/ directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def collect_games_for_date(date_str: str) -> list[int]:
    """
    Get all completed game IDs for a specific date.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        List of game IDs that have finished (gameState == 'OFF').
    """
    schedule = nhl_api.get_schedule(date_str)
    game_ids = []

    for day in schedule.get("gameWeek", []):
        if day["date"] != date_str:
            continue
        for game in day.get("games", []):
            if game["gameState"] == "OFF":
                game_ids.append(game["id"])

    return game_ids


def collect_date_range(
    start_date: str,
    end_date: str,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Collect player-level boxscore data for a range of dates.

    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        delay: Seconds to wait between API calls (be nice to the API!)

    Returns:
        DataFrame with player-game stats for all games in the range.

    💡 CONCEPT: We add a delay between requests because hammering
    an API with rapid-fire requests is rude (and might get us
    rate-limited or blocked). This is called "rate limiting."
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_frames = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"📅 Fetching games for {date_str}...")

        try:
            game_ids = collect_games_for_date(date_str)
            print(f"   Found {len(game_ids)} completed games")

            for gid in game_ids:
                try:
                    df = get_game_player_stats(gid)
                    all_frames.append(df)
                    time.sleep(delay)
                except nhl_api.NHLApiError as e:
                    print(f"   ⚠️  Failed boxscore for game {gid}: {e}")

        except nhl_api.NHLApiError as e:
            print(f"   ⚠️  Failed schedule for {date_str}: {e}")

        current += timedelta(days=1)

    if not all_frames:
        print("No data collected!")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n✅ Collected {len(combined)} player-game rows")
    return combined


def save_game_data(df: pd.DataFrame, filename: str = "game_log.csv"):
    """
    Save collected data to CSV in the data/ directory.

    Args:
        df: DataFrame to save.
        filename: Output filename.
    """
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"💾 Saved {len(df)} rows to {path}")


def load_game_data(filename: str = "game_log.csv") -> pd.DataFrame:
    """
    Load previously saved game data from CSV.

    Args:
        filename: CSV filename in data/ directory.

    Returns:
        DataFrame of player-game stats.

    Raises:
        FileNotFoundError: If no saved data exists yet.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at {path}. Run collect_date_range() first!"
        )
    return pd.read_csv(path)
