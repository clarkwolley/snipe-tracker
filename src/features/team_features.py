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
]
