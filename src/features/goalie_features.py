"""
Goalie matchup features for the goal scorer model.

💡 KEY CONCEPT: The opposing goalie matters A LOT. A player facing
a .880 save% backup has a way better shot at scoring than one facing
a .925 starter. This module pulls goalie stats and exposes them
as features for our prediction pipeline.

We focus on the starter (most games played) since we can't always
know who's starting tonight from the free API alone.
"""

import pandas as pd

from src.data.collector import get_team_goalies


def get_likely_starter(team_abbrev: str) -> dict:
    """
    Get stats for the team's likely starter (most GP this season).

    Args:
        team_abbrev: Three-letter team code (e.g., 'COL')

    Returns:
        Dict with goalie stats, or defaults if unavailable.

    💡 CONCEPT: Without confirmed lineup data, we approximate by
    picking the goalie with the most games played — the "1A".
    This is right ~65-70% of the time. Good enough for a feature.
    """
    try:
        goalies = get_team_goalies(team_abbrev)
    except Exception:
        return _default_goalie_stats()

    if goalies.empty:
        return _default_goalie_stats()

    starter = goalies.sort_values("games_played", ascending=False).iloc[0]

    return {
        "goalie_id": starter["player_id"],
        "goalie_name": f"{starter['first_name']} {starter['last_name']}",
        "goalie_gp": starter["games_played"],
        "goalie_save_pct": starter["save_pct"],
        "goalie_gaa": starter["goals_against_avg"],
        "goalie_wins": starter["wins"],
        "goalie_shutouts": starter["shutouts"],
    }


def _default_goalie_stats() -> dict:
    """League-average fallback when goalie data is unavailable."""
    return {
        "goalie_id": 0,
        "goalie_name": "Unknown",
        "goalie_gp": 0,
        "goalie_save_pct": 0.900,
        "goalie_gaa": 3.00,
        "goalie_wins": 0,
        "goalie_shutouts": 0,
    }


def build_goalie_matchup_features(opponent_team: str) -> dict:
    """
    Build goalie-based features for a matchup.

    Args:
        opponent_team: The OPPOSING team abbreviation (the team whose
                       goalie our skaters are shooting against).

    Returns:
        Dict with goalie matchup features:
        - opp_goalie_save_pct: Opposing goalie's save percentage
        - opp_goalie_gaa: Opposing goalie's goals against average
        - opp_goalie_quality: Composite quality score (higher = tougher)

    💡 CONCEPT: We invert the goalie perspective — a HIGH save%
    goalie is BAD for the skater trying to score. The quality
    score combines save% and GAA into a single "how hard is it
    to score on this goalie" metric.
    """
    starter = get_likely_starter(opponent_team)

    save_pct = starter["goalie_save_pct"]
    gaa = starter["goalie_gaa"]

    # Quality score: normalize save% (0-1) and inverse GAA
    # Higher = tougher goalie to score on
    quality = (save_pct * 10) - (gaa * 0.5)  # rough composite

    return {
        "opp_goalie_save_pct": save_pct,
        "opp_goalie_gaa": gaa,
        "opp_goalie_quality": round(quality, 3),
        "opp_goalie_name": starter["goalie_name"],
    }
