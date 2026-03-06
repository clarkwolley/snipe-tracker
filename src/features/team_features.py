"""
Team-level feature engineering for the game winner model.

💡 KEY CONCEPT: While player features ask "will THIS player score?",
team features ask "will THIS team win?" Different question,
different features.

Team features focus on:
- Overall team strength (point %, goal differential)
- Home vs. away performance (home ice advantage is real!)
- Matchup dynamics (offense vs. defense ratings)
"""

import pandas as pd
import numpy as np


def build_team_strength(standings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team strength metrics from standings.

    New columns:
    - goals_for_pg: Goals scored per game
    - goals_against_pg: Goals allowed per game
    - goal_diff_pg: Goal differential per game
    - home_win_pct: Win % at home
    - road_win_pct: Win % on the road
    - pp_pct: Power play percentage (if available from standings)
    - pk_pct: Penalty kill percentage (if available from standings)

    💡 CONCEPT: Per-game rates let us compare teams fairly even if
    they've played different numbers of games. A team scoring 3.5
    goals/game is more dangerous than one at 2.5, regardless of
    how many games each has played.
    """
    df = standings_df.copy()
    gp = df["games_played"].clip(lower=1)  # avoid division by zero

    df["goals_for_pg"] = df["goals_for"] / gp
    df["goals_against_pg"] = df["goals_against"] / gp
    df["goal_diff_pg"] = df["goal_diff"] / gp

    home_games = (df["home_wins"] + df["home_losses"]).clip(lower=1)
    road_games = (df["road_wins"] + df["road_losses"]).clip(lower=1)

    df["home_win_pct"] = df["home_wins"] / home_games
    df["road_win_pct"] = df["road_wins"] / road_games

    # Special teams — use standings data if present, else league avg
    if "pp_pct" not in df.columns:
        df["pp_pct"] = 0.20  # ~20% league average
    if "pk_pct" not in df.columns:
        df["pk_pct"] = 0.80  # ~80% league average

    return df


def build_matchup_features(
    home_team: str,
    away_team: str,
    team_strength: pd.DataFrame,
) -> dict:
    """
    Build features for a specific home vs. away matchup.

    Args:
        home_team: Home team abbreviation (e.g., 'COL')
        away_team: Away team abbreviation (e.g., 'TOR')
        team_strength: DataFrame from build_team_strength()

    Returns:
        Dict of matchup features ready for model input.

    💡 KEY CONCEPT: Matchup features capture the RELATIVE strength
    between two teams. A great team vs. a great team is different
    from a great team vs. a bad team. The model needs to know the
    gap, not just absolute strength.

    We prefix features with 'home_' and 'away_' so the model can
    learn that being the home team matters (home ice advantage
    in hockey is roughly +4% win probability).
    """
    home = team_strength[team_strength["team"] == home_team]
    away = team_strength[team_strength["team"] == away_team]

    if home.empty or away.empty:
        return {}

    home = home.iloc[0]
    away = away.iloc[0]

    return {
        # Home team stats
        "home_point_pct": home["point_pct"],
        "home_goals_for_pg": home["goals_for_pg"],
        "home_goals_against_pg": home["goals_against_pg"],
        "home_goal_diff_pg": home["goal_diff_pg"],
        "home_home_win_pct": home["home_win_pct"],
        # Away team stats
        "away_point_pct": away["point_pct"],
        "away_goals_for_pg": away["goals_for_pg"],
        "away_goals_against_pg": away["goals_against_pg"],
        "away_goal_diff_pg": away["goal_diff_pg"],
        "away_road_win_pct": away["road_win_pct"],
        # Matchup differentials (home advantage)
        "point_pct_diff": home["point_pct"] - away["point_pct"],
        "goal_diff_pg_diff": home["goal_diff_pg"] - away["goal_diff_pg"],
        "offense_vs_defense": home["goals_for_pg"] - away["goals_against_pg"],
        "defense_vs_offense": away["goals_for_pg"] - home["goals_against_pg"],
        # Special teams
        "home_pp_pct": home.get("pp_pct", 0.20),
        "home_pk_pct": home.get("pk_pct", 0.80),
        "away_pp_pct": away.get("pp_pct", 0.20),
        "away_pk_pct": away.get("pk_pct", 0.80),
        # PP vs PK matchup: home PP attacking away PK and vice versa
        "home_pp_vs_away_pk": home.get("pp_pct", 0.20) - (1 - away.get("pk_pct", 0.80)),
        "away_pp_vs_home_pk": away.get("pp_pct", 0.20) - (1 - home.get("pk_pct", 0.80)),
    }


def build_rest_features(
    home_team: str,
    away_team: str,
    game_date: str,
) -> dict:
    """
    Build fatigue/rest features for a matchup.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        game_date: Game date 'YYYY-MM-DD'

    Returns:
        Dict with rest features:
        - home_b2b: 1 if home team on back-to-back, 0 otherwise
        - away_b2b: 1 if away team on back-to-back, 0 otherwise
        - home_days_rest: Days since home team's last game
        - away_days_rest: Days since away team's last game
        - rest_advantage: Positive = home team more rested

    💡 CONCEPT: Fatigue is real. Teams on a back-to-back lose
    ~4-5% more often than rested teams. The effect is even
    larger for the away team on a B2B (travel fatigue stacks).
    """
    from src.data.collector import get_back_to_back_status

    try:
        home_rest = get_back_to_back_status(home_team, game_date)
        away_rest = get_back_to_back_status(away_team, game_date)
    except Exception:
        return {
            "home_b2b": 0, "away_b2b": 0,
            "home_days_rest": 2, "away_days_rest": 2,
            "rest_advantage": 0,
        }

    return {
        "home_b2b": int(home_rest["is_back_to_back"]),
        "away_b2b": int(away_rest["is_back_to_back"]),
        "home_days_rest": home_rest["days_rest"],
        "away_days_rest": away_rest["days_rest"],
        "rest_advantage": away_rest["days_rest"] - home_rest["days_rest"],
    }


def build_game_features(
    schedule_df: pd.DataFrame,
    team_strength: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features for all games in a schedule.

    Args:
        schedule_df: DataFrame with 'home_team' and 'away_team' columns
        team_strength: DataFrame from build_team_strength()

    Returns:
        DataFrame with one row per game, all matchup features added.
    """
    rows = []
    for _, game in schedule_df.iterrows():
        features = build_matchup_features(
            game["home_team"],
            game["away_team"],
            team_strength,
        )
        features["game_id"] = game.get("game_id", "")
        features["home_team"] = game["home_team"]
        features["away_team"] = game["away_team"]
        rows.append(features)

    return pd.DataFrame(rows)


# Columns our game winner model will use
GAME_FEATURE_COLUMNS = [
    "home_point_pct",
    "home_goals_for_pg",
    "home_goals_against_pg",
    "home_goal_diff_pg",
    "home_home_win_pct",
    "away_point_pct",
    "away_goals_for_pg",
    "away_goals_against_pg",
    "away_goal_diff_pg",
    "away_road_win_pct",
    "point_pct_diff",
    "goal_diff_pg_diff",
    "offense_vs_defense",
    "defense_vs_offense",
    # Special teams
    "home_pp_pct",
    "home_pk_pct",
    "away_pp_pct",
    "away_pk_pct",
    "home_pp_vs_away_pk",
    "away_pp_vs_home_pk",
]
