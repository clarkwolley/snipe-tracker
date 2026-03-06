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
from src.config import ROLLING_WINDOW, MIN_GAMES_PLAYED, STREAK_MIN_GAMES


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


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect hot streaks and cold slumps for each player.

    New columns:
    - goal_streak: Consecutive games with a goal (0 if none)
    - point_streak: Consecutive games with a point (0 if none)
    - drought: Consecutive games WITHOUT a goal (0 if scored last game)
    - is_hot: Binary flag — on a goal streak of 2+ games

    💡 KEY CONCEPT: Streaks capture momentum. A player who scored
    in 4 straight games is "seeing the puck" — their confidence
    and shot selection tend to be better. Sports are as much
    psychology as they are physics.
    """
    result = df.copy()
    result = result.sort_values(["player_id", "game_date"])

    def _calc_streak(series: pd.Series) -> pd.Series:
        """Count consecutive True values ending at each position."""
        streaks = []
        count = 0
        for val in series:
            if val:
                count += 1
            else:
                count = 0
            streaks.append(count)
        return pd.Series(streaks, index=series.index)

    def _calc_drought(series: pd.Series) -> pd.Series:
        """Count consecutive False values (no goals) ending at each position."""
        droughts = []
        count = 0
        for val in series:
            if not val:
                count += 1
            else:
                count = 0
            droughts.append(count)
        return pd.Series(droughts, index=series.index)

    # Ensure 'scored' column exists
    if "scored" not in result.columns:
        result["scored"] = (result["goals"] > 0).astype(int)

    scored_bool = result["scored"].astype(bool)
    has_point = (result["points"] > 0) if "points" in result.columns else scored_bool

    # Goal streak: shift by 1 so we see the streak ENTERING the game
    result["goal_streak"] = (
        result.groupby("player_id")["scored"]
        .transform(lambda x: _calc_streak(x.astype(bool)).shift(1).fillna(0))
        .astype(int)
    )

    result["point_streak"] = (
        result.groupby("player_id")[has_point.name if hasattr(has_point, 'name') else "points"]
        .transform(lambda x: _calc_streak((x > 0) if x.dtype != bool else x).shift(1).fillna(0))
        .astype(int)
    )

    result["drought"] = (
        result.groupby("player_id")["scored"]
        .transform(lambda x: _calc_drought(x.astype(bool)).shift(1).fillna(0))
        .astype(int)
    )

    result["is_hot"] = (result["goal_streak"] >= STREAK_MIN_GAMES).astype(int)

    return result


def add_shot_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shot volume and efficiency features.

    New columns:
    - shots_per_toi: Shots per minute of ice time (shot generation rate)
    - shooting_pct_trend: Change in shooting % over rolling window
    - high_volume_shooter: Binary flag — above-average shot volume

    💡 CONCEPT: Raw shot counts are misleading — a 4th liner who
    plays 8 minutes and gets 1 shot isn't comparable to a 1st liner
    who plays 22 minutes and gets 4 shots. Normalizing by TOI gives
    us the TRUE shot generation rate.

    "High danger" in the NHL world means shots from the slot area.
    We can't get exact location data from the free API, but shot RATE
    is a decent proxy — players who generate lots of shots per minute
    tend to be getting quality chances from dangerous areas.
    """
    result = df.copy()

    # Ensure toi_minutes exists
    if "toi_minutes" not in result.columns:
        result["toi_minutes"] = result["toi"].apply(parse_toi_to_minutes)

    # Shots per minute of ice time (shot generation rate)
    result["shots_per_toi"] = np.where(
        result["toi_minutes"] > 0,
        result["shots"] / result["toi_minutes"],
        0.0,
    )

    # Rolling shooting % trend: current rolling vs. season average
    if "rolling_shooting_pct" in result.columns:
        season_avg_pct = result.groupby("player_id")["rolling_shooting_pct"].transform("mean")
        result["shooting_pct_trend"] = result["rolling_shooting_pct"] - season_avg_pct
    else:
        result["shooting_pct_trend"] = 0.0

    # High volume shooter flag
    result["high_volume_shooter"] = (
        result.groupby("player_id")["shots"]
        .transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean())
    )
    league_avg_shots = result["high_volume_shooter"].median()
    result["high_volume_shooter"] = (
        result["high_volume_shooter"] > league_avg_shots
    ).astype(int)

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

    # Step 4: Streak detection (hot hand / cold slumps)
    df = add_streak_features(df)

    # Step 5: Shot quality and volume features
    df = add_shot_quality_features(df)

    # Step 6: Position encoding
    df = add_position_encoding(df)

    # Step 7: Home/away as numeric
    df["is_home"] = df["is_home"].astype(int)

    return df


# The columns our model will actually use as inputs
FEATURE_COLUMNS = [
    # Rolling averages (recent form)
    "rolling_goals_avg",
    "rolling_shots_avg",
    "rolling_points_avg",
    "rolling_toi_avg",
    "rolling_shooting_pct",
    "games_in_window",
    # Streak features (momentum)
    "goal_streak",
    "point_streak",
    "drought",
    "is_hot",
    # Shot quality features
    "shots_per_toi",
    "shooting_pct_trend",
    "high_volume_shooter",
    # Player attributes
    "is_forward",
    "is_center",
    "is_home",
    # Goalie matchup features (injected at prediction time)
    "opp_goalie_save_pct",
    "opp_goalie_gaa",
    "opp_goalie_quality",
]

# What we're predicting
TARGET_COLUMN = "scored"
