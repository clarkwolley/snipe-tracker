"""
Historical game results collector for the game winner model.

💡 KEY CONCEPT: For the goal model, we needed player-level data.
For the game winner model, we need GAME-level outcomes:
"Team A played Team B at home and won 4-2."

We combine schedule data (who played who) with boxscore data
(who won) to build a training set of game outcomes.
"""

import time

import pandas as pd

from src.data import nhl_api


def get_game_result(game_id: int) -> dict | None:
    """
    Extract the outcome of a single game from its boxscore.

    Returns:
        Dict with game result info, or None if game isn't finished.
    """
    try:
        box = nhl_api.get_boxscore(game_id)
    except Exception:
        return None

    if box.get("gameState") != "OFF":
        return None

    home = box.get("homeTeam", {})
    away = box.get("awayTeam", {})

    home_score = home.get("score", 0)
    away_score = away.get("score", 0)

    return {
        "game_id": game_id,
        "game_date": box.get("gameDate", ""),
        "home_team": home.get("abbrev", ""),
        "away_team": away.get("abbrev", ""),
        "home_score": home_score,
        "away_score": away_score,
        "home_win": int(home_score > away_score),
        "total_goals": home_score + away_score,
    }


def collect_game_results(start_date: str, end_date: str, delay: float = 0.3) -> pd.DataFrame:
    """
    Collect game outcomes for a date range.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        delay: Seconds between API calls

    Returns:
        DataFrame with one row per game (home_team, away_team, home_win, etc.)
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    results = []

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            schedule = nhl_api.get_schedule(date_str)
            for day in schedule.get("gameWeek", []):
                if day["date"] != date_str:
                    continue
                for game in day.get("games", []):
                    if game["gameState"] != "OFF":
                        continue
                    result = get_game_result(game["id"])
                    if result:
                        results.append(result)
                    time.sleep(delay)
        except Exception as e:
            print(f"  ⚠️  {date_str}: {e}")

        current += timedelta(days=1)

    df = pd.DataFrame(results)
    print(f"✅ Collected {len(df)} game results")
    return df
