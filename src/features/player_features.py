"""
Player-level feature engineering for the goal scorer model.

💡 KEY CONCEPT: "Features" are the inputs to our model — the numbers
it uses to make predictions. Raw stats like "30 goals this season"
aren't great features because they don't account for games played.

Instead, we engineer RATES and TRENDS:
- Goals per game (rate) > total goals (raw)
- Last 5 games avg (trend) > season average (static)
- Shooting % (efficiency) > total shots (volume)

Good features capture WHY a player might score tonight.
"""

import pandas as pd
import numpy as np
from src.config import ROLLING_WINDOW, MIN_GAMES_PLAYED


def parse_toi_to_minutes(toi_str: str) -> float:
    """
    Convert time-on-ice string 'MM:SS' to decimal minutes.

    Examples:
        '20:30' → 20.5
        '15:00' → 15.0

    💡 CONCEPT: TOI is stored as a string like '18:45' but our model
    needs numbers. This kind of data cleaning is a huge part of
    feature engineering — real-world data is always messy.
    """
    if pd.isna(toi_str) or toi_str == "0:00":
        return 0.0
    try:
        parts = str(toi_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except (ValueError, IndexError):
        return 0.0


def add_basic_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-game rate features from season-level stats.

    New columns:
    - goals_per_game, shots_per_game, points_per_game
    - toi_minutes (parsed from string)
    - scored (binary: did the player score? Our prediction target!)

    💡 CONCEPT: Rates normalize for games played. A player with
    10 goals in 20 games (0.5/game) is more dangerous than a
    player with 15 goals in 60 games (0.25/game).
    """
    result = df.copy()
    result["toi_minutes"] = result["toi"].apply(parse_toi_to_minutes)
    result["scored"] = (result["goals"] > 0).astype(int)
    return result


def add_rolling_averages(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Calculate rolling averages for each player over their last N games.

    New columns:
    - rolling_goals_avg: Average goals over last N games
    - rolling_shots_avg: Average shots over last N games
    - rolling_points_avg: Average points over last N games
    - rolling_toi_avg: Average TOI over last N games
    - games_in_window: How many of the last N games we have data for

    💡 KEY CONCEPT: Rolling averages capture "recent form." A player
    on a hot streak (3 goals in last 5 games) is more likely to score
    than their season average suggests. This is one of the most
    powerful features in sports prediction.

    The "window" is how many games to look back. Too small (2-3) and
    it's noisy. Too large (20+) and it's just the season average.
    We default to 10 as a good balance.
    """
    result = df.copy()
    result = result.sort_values(["player_id", "game_date"])

    rolling_cols = {
        "goals": "rolling_goals_avg",
        "shots": "rolling_shots_avg",
        "points": "rolling_points_avg",
        "toi_minutes": "rolling_toi_avg",
    }

    # Need toi_minutes first
    if "toi_minutes" not in result.columns:
        result["toi_minutes"] = result["toi"].apply(parse_toi_to_minutes)

    for raw_col, new_col in rolling_cols.items():
        result[new_col] = (
            result
            .groupby("player_id")[raw_col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    # Track how many games in the rolling window (more data = more reliable)
    result["games_in_window"] = (
        result
        .groupby("player_id")["goals"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).count())
    )

    return result


def add_shooting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shooting efficiency features.

    New columns:
    - rolling_shooting_pct: Goals / shots over the rolling window
    - shots_above_avg: How many more shots than their own average

    💡 CONCEPT: Shooting percentage tells us about quality vs quantity.
    A player getting 5 shots/game with 15% shooting is more likely
    to score than one getting 2 shots/game at 8%.
    """
    result = df.copy()

    if "rolling_goals_avg" not in result.columns or "rolling_shots_avg" not in result.columns:
        result = add_rolling_averages(result)

    # Rolling shooting percentage (avoid division by zero)
    result["rolling_shooting_pct"] = np.where(
        result["rolling_shots_avg"] > 0,
        result["rolling_goals_avg"] / result["rolling_shots_avg"],
        0.0,
    )

    # Shots above their own rolling average (hot hand indicator)
    player_avg_shots = result.groupby("player_id")["shots"].transform("mean")
    result["shots_above_avg"] = result["shots"] - player_avg_shots

    return result


def add_position_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode player position as numeric features.

    New columns:
    - is_forward: 1 if C/L/R, 0 if D
    - is_center: 1 if C, 0 otherwise

    💡 CONCEPT: Forwards score way more than defensemen (~75% of goals).
    Position is a strong baseline signal. We encode it as numbers
    because ML models need numbers, not strings like 'C' or 'D'.
    """
    result = df.copy()
    result["is_forward"] = result["position"].isin(["C", "L", "R"]).astype(int)
    result["is_center"] = (result["position"] == "C").astype(int)
    return result


def build_player_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for player-level prediction.

    Takes a raw game log and returns a feature-rich DataFrame ready
    for model training.

    Args:
        game_log: Raw game log from data collection (game_log.csv)

    Returns:
        DataFrame with all engineered features added.

    💡 CONCEPT: This function chains all the individual feature
    functions together into a "pipeline." Each step adds new columns.
    By keeping each step as a separate function, we can test them
    individually and mix-and-match as needed.
    """
    df = game_log.copy()

    # Step 1: Basic rates and target variable
    df = add_basic_rates(df)

    # Step 2: Rolling averages (recent form)
    df = add_rolling_averages(df)

    # Step 3: Shooting features
    df = add_shooting_features(df)

    # Step 4: Position encoding
    df = add_position_encoding(df)

    # Step 5: Home/away as numeric
    df["is_home"] = df["is_home"].astype(int)

    return df


# The columns our model will actually use as inputs
FEATURE_COLUMNS = [
    "rolling_goals_avg",
    "rolling_shots_avg",
    "rolling_points_avg",
    "rolling_toi_avg",
    "rolling_shooting_pct",
    "games_in_window",
    "is_forward",
    "is_center",
    "is_home",
]

# What we're predicting
TARGET_COLUMN = "scored"
