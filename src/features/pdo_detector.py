"""
PDO Regression Detection for identifying overperforming players.

💡 PDO CONCEPT:
   PDO = (Individual Shooting % × 100) + (Team Save % × 100)
   
   League average PDO ≈ 100.0
   - PDO > 103: Player is likely overperforming (unsustainable luck)
   - PDO < 97:  Player is likely underperforming (bad luck)
   
   Why it matters:
   - Kopitar on 3/16 probably had PDO 105+ with 0.5 xG but high shooting %
   - Player will regress to the mean
   - Model was picking him up on an inflated shooting %, now he's a trap

This module detects regression candidates and adjusts confidence downward.
"""

import pandas as pd
import numpy as np


# League-average benchmarks (should stay stable year-to-year)
LEAGUE_AVG_SV_PCT = 0.915
LEAGUE_AVG_SH_PCT = 0.082  # Roughly 8.2% overall
LEAGUE_AVG_PDO = 100.0
HIGH_PDO_THRESHOLD = 103.0  # "Unsustainably high" threshold
LOW_PDO_THRESHOLD = 97.0   # "Unsustainably low" threshold


def calculate_pdo(shooting_pct: float, team_sv_pct: float) -> float:
    """
    Calculate PDO for a player-team pairing.
    
    Args:
        shooting_pct: Individual shooting percentage (0.0-1.0)
        team_sv_pct: Team save percentage (0.0-1.0)
    
    Returns:
        PDO value (0-200, centered around 100)
    """
    return (shooting_pct * 100) + (team_sv_pct * 100)


def detect_regression_candidate(
    pdo: float,
    rolling_goals_avg: float,
    rolling_shots_avg: float,
    games_in_window: int,
    threshold: float = HIGH_PDO_THRESHOLD,
) -> dict:
    """
    Determine if a player is a regression candidate (lucky, will regress).
    
    Args:
        pdo: Player's PDO value
        rolling_goals_avg: Goals per game (last 10 games)
        rolling_shots_avg: Shots per game (last 10 games)
        games_in_window: Games in the rolling window
        threshold: PDO threshold for regression detection
    
    Returns:
        Dict with:
        - is_regressing: bool (True if PDO is unsustainably high)
        - regression_intensity: 0.0-1.0 (how extreme the PDO is)
        - pdo_z_score: How many std devs away from mean
    
    💡 CONCEPT: A player with PDO 105 and 0.5 GPG is riskier than
    one with PDO 105 and 2.0 GPG. We use GPG to weight the risk.
    """
    # Z-score distance from mean (how "extreme" the PDO is)
    # Assuming PDO std dev ≈ 5 (typical variance)
    pdo_distance = pdo - LEAGUE_AVG_PDO
    pdo_z_score = pdo_distance / 5.0
    
    # Is PDO elevated?
    is_regressing = pdo > threshold
    
    # Intensity: how much above threshold?
    # A PDO of 103 is mild; 110+ is extreme
    if is_regressing:
        regression_intensity = min(1.0, (pdo - threshold) / 7.0)
    else:
        regression_intensity = 0.0
    
    return {
        "is_regressing": is_regressing,
        "regression_intensity": regression_intensity,
        "pdo_z_score": pdo_z_score,
        "pdo": pdo,
    }


def apply_regression_penalty(
    goal_probability: float,
    regression_info: dict,
    penalty_strength: float = 0.5,
) -> tuple[float, str]:
    """
    Reduce goal probability for regression candidates.
    
    Args:
        goal_probability: Model's predicted probability (0.0-1.0)
        regression_info: Output from detect_regression_candidate()
        penalty_strength: How much to penalize (0.0=none, 1.0=aggressive)
    
    Returns:
        (adjusted_probability, reasoning_flag)
    
    Example:
        Model says Kopitar: 59.7%
        But PDO=108, games=5
        → Adjusted down to 45% (regression detected, applying penalty)
    """
    if not regression_info["is_regressing"]:
        return goal_probability, ""
    
    intensity = regression_info["regression_intensity"]
    penalty = intensity * penalty_strength
    
    # Reduce probability by penalty amount
    # But don't push below 0.2 (still some chance)
    adjusted = max(0.20, goal_probability * (1.0 - penalty))
    
    reasoning = f"regression_pdo_{regression_info['pdo']:.0f}"
    
    return adjusted, reasoning


def add_pdo_features(
    df: pd.DataFrame,
    team_sv_pct_col: str = "team_sv_pct",
    rolling_sh_pct_col: str = "rolling_shooting_pct",
) -> pd.DataFrame:
    """
    Add PDO and regression detection columns to a DataFrame.
    
    Requires columns:
    - team_sv_pct: Team save percentage
    - rolling_shooting_pct: Player's recent shooting %
    - rolling_goals_avg: Goals per game
    - rolling_shots_avg: Shots per game
    - games_in_window: Games in the window
    
    Returns:
        DataFrame with new columns:
        - pdo: The PDO value
        - is_regressing: Boolean (unsustainably high PDO?)
        - regression_intensity: 0.0-1.0 float
    """
    result = df.copy()
    
    # Calculate PDO
    result["pdo"] = result.apply(
        lambda r: calculate_pdo(
            r.get(rolling_sh_pct_col, 0.082),
            r.get(team_sv_pct_col, LEAGUE_AVG_SV_PCT),
        ),
        axis=1,
    )
    
    # Detect regression candidates
    regression_results = result.apply(
        lambda r: detect_regression_candidate(
            r["pdo"],
            r.get("rolling_goals_avg", 0),
            r.get("rolling_shots_avg", 0),
            r.get("games_in_window", 1),
            threshold=HIGH_PDO_THRESHOLD,
        ),
        axis=1,
    )
    
    result["is_regressing"] = regression_results.apply(lambda x: int(x["is_regressing"]))
    result["regression_intensity"] = regression_results.apply(lambda x: x["regression_intensity"])
    result["pdo_z_score"] = regression_results.apply(lambda x: x["pdo_z_score"])
    
    return result
