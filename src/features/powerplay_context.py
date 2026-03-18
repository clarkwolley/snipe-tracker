"""
Power Play context features — helps the model understand PP vs EV environment.

💡 WHY IT MATTERS:
   - PP goals are ~3x more predictable than EV goals
   - Some players live on the PP (specialists)
   - Model needs to know: "Is this a PP-heavy night?"
   
   Signals:
   - Is the player a PP specialist? (high PP goal %)
   - Does the opponent PK suck? (high PP against rate)
   - Is the game expected to be high-PP? (certain matchups)
"""

import pandas as pd


def is_pp_specialist(
    rolling_pp_goals_avg: float,
    rolling_goals_avg: float,
    min_pp_threshold: float = 0.15,
) -> int:
    """
    Detect if a player is a power play specialist.
    
    Args:
        rolling_pp_goals_avg: PP goals per game (last 10 games)
        rolling_goals_avg: Total goals per game (last 10 games)
        min_pp_threshold: Min % of goals from PP (15% = specialist)
    
    Returns:
        1 if specialist, 0 otherwise
    
    Example:
        Player scores 1.0/game total, 0.3/game on PP
        → 30% from PP → specialist (above 15% threshold)
    """
    if rolling_goals_avg <= 0:
        return 0
    
    pp_goal_share = rolling_pp_goals_avg / rolling_goals_avg
    return int(pp_goal_share >= min_pp_threshold)


def get_pp_context_for_game(
    team_pp_pct: float,
    opp_pk_pct: float = None,
) -> dict:
    """
    Assess PP environment for a specific game.
    
    Args:
        team_pp_pct: Team's PP conversion % for the season
        opp_pk_pct: Opponent's PK % (if available)
    
    Returns:
        Dict with:
        - team_pp_strength: 0.0-1.0 (is our PP good?)
        - opp_pk_weakness: 0.0-1.0 (can we score on them?)
        - pp_environment: 'strong'/'neutral'/'weak'
    
    💡 CONCEPT: A PP that scores 25%+ is "elite". A team that kills
    90%+ of PPs is strong on defense. We can signal this to the model.
    """
    league_avg_pp = 0.20  # ~20% PP conversion league-wide
    
    team_pp_strength = min(1.0, team_pp_pct / 0.25)  # Normalized to 25% elite
    
    # PK percentage (% of PPs killed) inversely relates to scoring
    # High PK% = hard to score on them
    if opp_pk_pct is not None:
        # Flip it: 90% PK = 10% PP allowed
        opp_pp_allowed = 1.0 - opp_pk_pct
        opp_pk_weakness = opp_pp_allowed / league_avg_pp
    else:
        # No data, assume neutral
        opp_pk_weakness = 1.0
    
    # Classify environment
    pp_env = team_pp_strength * opp_pk_weakness
    if pp_env > 1.2:
        environment = "strong"
    elif pp_env < 0.8:
        environment = "weak"
    else:
        environment = "neutral"
    
    return {
        "team_pp_strength": team_pp_strength,
        "opp_pk_weakness": opp_pk_weakness,
        "pp_environment": environment,
    }


def add_pp_context_features(
    df: pd.DataFrame,
    team_pp_pct_col: str = "team_pp_pct",
) -> pd.DataFrame:
    """
    Add power play context columns to prediction DataFrame.
    
    Requires:
    - rolling_pp_goals_avg: PP goals/game
    - rolling_goals_avg: Total goals/game
    - team_pp_pct: Team's season PP % (optional)
    
    Adds:
    - is_pp_specialist: Binary flag
    - pp_dependence: Share of goals from PP (0.0-1.0)
    """
    result = df.copy()
    
    # PP specialist detection
    result["is_pp_specialist"] = result.apply(
        lambda r: is_pp_specialist(
            r.get("rolling_pp_goals_avg", 0),
            r.get("rolling_goals_avg", 0),
        ),
        axis=1,
    )
    
    # How much of their scoring comes from PP?
    result["pp_dependence"] = result.apply(
        lambda r: (
            r.get("rolling_pp_goals_avg", 0) / max(r.get("rolling_goals_avg", 1), 0.1)
            if r.get("rolling_goals_avg", 0) > 0 else 0.0
        ),
        axis=1,
    ).clip(0, 1.0)  # Bound between 0 and 1
    
    return result
