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
from src.config import ROLLING_WINDOW, RECENCY_WINDOW, RECENCY_WEIGHT, STREAK_MIN_GAMES


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



def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling averages with recency weighting.

    Uses a 50-game window with two weighting strategies:

    **Volume metrics** (shots, goals, points, TOI, PP goals):
      The most recent 10 games are weighted 1.5x.  Volume reflects
      opportunity and deployment — if a coach just moved a player to
      the top line last week, the recent 10 games matter more than
      the 40 before that.

    **Efficiency metrics** (shooting %):
      Equal weight across the full 50 games.  Larger samples give a
      truer read on a player's conversion rate; small-sample SH%
      is notoriously noisy.

    New columns:
    - rolling_goals_avg, rolling_shots_avg, rolling_points_avg,
      rolling_toi_avg, rolling_pp_goals_avg  (recency-weighted volume)
    - rolling_shooting_pct  (50-game equal-weight efficiency)
    - games_in_window
    """
    result = df.copy()
    result = result.sort_values(["player_id", "game_date"])

    if "toi_minutes" not in result.columns:
        result["toi_minutes"] = result["toi"].apply(parse_toi_to_minutes)
    if "pp_goals" not in result.columns:
        result["pp_goals"] = 0

    # --- helper: shifted rolling aggregation per player ----------------
    def _rolling(col: str, window: int, agg: str = "mean") -> pd.Series:
        return (
            result
            .groupby("player_id")[col]
            .transform(
                lambda x: getattr(
                    x.shift(1).rolling(window, min_periods=1), agg
                )()
            )
        )

    # Games in full window (shared denominator)
    count_full = _rolling("goals", ROLLING_WINDOW, "count")

    # --- Volume metrics: recency-weighted ------------------------------
    #
    # Formula:
    #   n_recent  = min(games_in_window, RECENCY_WINDOW)
    #   boost     = RECENCY_WEIGHT - 1.0          (0.5 with defaults)
    #   weight    = count_full + n_recent * boost  (total weight)
    #   weighted  = (roll_full * count_full
    #                + roll_recent * n_recent * boost) / weight
    #
    # When a player has <= RECENCY_WINDOW games, all games are
    # "recent" and the formula collapses to a simple mean.

    volume_cols = {
        "goals":       "rolling_goals_avg",
        "shots":       "rolling_shots_avg",
        "points":      "rolling_points_avg",
        "toi_minutes": "rolling_toi_avg",
        "pp_goals":    "rolling_pp_goals_avg",
    }

    boost = RECENCY_WEIGHT - 1.0  # 0.5 with default config
    n_recent = count_full.clip(upper=RECENCY_WINDOW)

    for raw_col, new_col in volume_cols.items():
        roll_full   = _rolling(raw_col, ROLLING_WINDOW, "mean")
        roll_recent = _rolling(raw_col, RECENCY_WINDOW, "mean")

        total_weight = count_full + n_recent * boost

        result[new_col] = np.where(
            total_weight > 0,
            (roll_full * count_full + roll_recent * n_recent * boost)
            / total_weight,
            0.0,
        )

    # --- Efficiency metric: equal-weight over full window ---------------
    goals_sum = _rolling("goals", ROLLING_WINDOW, "sum")
    shots_sum = _rolling("shots", ROLLING_WINDOW, "sum")

    result["rolling_shooting_pct"] = np.where(
        shots_sum > 0,
        goals_sum / shots_sum,
        0.0,
    )

    result["games_in_window"] = count_full

    return result


def add_shooting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shooting-derived features.

    New columns:
    - shots_above_avg: How many more shots than their own average

    NOTE: rolling_shooting_pct is now computed inside
    add_rolling_averages() as an equal-weight efficiency metric
    (50-game window, no recency boost).  Keeping it there avoids
    accidentally deriving SH% from recency-weighted shot/goal
    averages, which would defeat the purpose.
    """
    result = df.copy()

    if "rolling_goals_avg" not in result.columns:
        result = add_rolling_averages(result)

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

    # Max gap between consecutive games before we reset streaks.
    # Catches season breaks (~5 months) but not all-star breaks (~1 week).
    MAX_STREAK_GAP_DAYS = 30

    def _calc_streak(scored: pd.Series, dates: pd.Series) -> pd.Series:
        """Count consecutive True values, resetting on season breaks."""
        streaks = []
        count = 0
        prev_date = None
        for val, date in zip(scored, dates):
            if prev_date is not None:
                gap = (date - prev_date).days
                if gap > MAX_STREAK_GAP_DAYS:
                    count = 0
            count = count + 1 if val else 0
            streaks.append(count)
            prev_date = date
        return pd.Series(streaks, index=scored.index)

    def _calc_drought(scored: pd.Series, dates: pd.Series) -> pd.Series:
        """Count consecutive False values, resetting on season breaks."""
        droughts = []
        count = 0
        prev_date = None
        for val, date in zip(scored, dates):
            if prev_date is not None:
                gap = (date - prev_date).days
                if gap > MAX_STREAK_GAP_DAYS:
                    count = 0
            count = count + 1 if not val else 0
            droughts.append(count)
            prev_date = date
        return pd.Series(droughts, index=scored.index)

    # Ensure 'scored' column exists
    if "scored" not in result.columns:
        result["scored"] = (result["goals"] > 0).astype(int)

    scored_bool = result["scored"].astype(bool)
    has_point = (result["points"] > 0) if "points" in result.columns else scored_bool

    # Parse dates once for gap detection
    dates = pd.to_datetime(result["game_date"])

    # Goal streak: shift by 1 so we see the streak ENTERING the game
    result["goal_streak"] = (
        result.groupby("player_id")
        .apply(lambda g: _calc_streak(g["scored"].astype(bool), dates.loc[g.index]).shift(1).fillna(0))
        .droplevel(0)
        .astype(int)
    )

    result["point_streak"] = (
        result.groupby("player_id")
        .apply(lambda g: _calc_streak(g["points"] > 0, dates.loc[g.index]).shift(1).fillna(0))
        .droplevel(0)
        .astype(int)
    )

    result["drought"] = (
        result.groupby("player_id")
        .apply(lambda g: _calc_drought(g["scored"].astype(bool), dates.loc[g.index]).shift(1).fillna(0))
        .droplevel(0)
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
    # Power play production (recency-weighted volume)
    "rolling_pp_goals_avg",
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
