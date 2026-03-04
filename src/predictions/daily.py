"""
Daily goal scorer predictions.

Pulls tonight's schedule, builds features for every skater playing,
runs the trained model, and ranks players by scoring probability.

💡 This is the payoff — everything we built (data pipeline, features,
model) comes together here to give you actual, actionable picks.
"""

import pandas as pd
import numpy as np
from tabulate import tabulate

from src.data import nhl_api
from src.data.collector import get_todays_games, get_team_skaters
from src.data.history import load_game_data
from src.features.player_features import (
    build_player_features,
    FEATURE_COLUMNS,
)
from src.models.goal_model import load_goal_model, predict_goal_probability


def _get_teams_playing_today() -> list[dict]:
    """
    Get all teams playing in today's earliest scheduled games.

    Returns:
        List of dicts with team abbreviation and home/away status.
    """
    schedule = get_todays_games()
    if schedule.empty:
        return []

    today = schedule[schedule["date"] == schedule["date"].min()]
    teams = []
    for _, game in today.iterrows():
        teams.append({"team": game["home_team"], "is_home": True,
                      "opponent": game["away_team"], "game_id": game["game_id"]})
        teams.append({"team": game["away_team"], "is_home": False,
                      "opponent": game["home_team"], "game_id": game["game_id"]})
    return teams


def _build_prediction_features(teams: list[dict], game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features for all players on tonight's teams.

    Strategy:
    1. Get season stats for each team's roster (current ability)
    2. Look up each player's recent game log (recent form)
    3. Combine into the features our model expects

    💡 CONCEPT: For predictions, we need the same features the model
    was trained on. The rolling averages come from the player's
    RECENT games — the last 10 games they played.
    """
    # Get recent game log for rolling averages
    featured = build_player_features(game_log)

    # For each player, get their LATEST rolling features
    latest = (
        featured
        .sort_values("game_date")
        .groupby("player_id")
        .last()
        .reset_index()
    )

    all_predictions = []

    for team_info in teams:
        team = team_info["team"]
        is_home = team_info["is_home"]
        opponent = team_info["opponent"]

        try:
            roster = get_team_skaters(team)
        except Exception as e:
            print(f"  ⚠️  Could not fetch roster for {team}: {e}")
            continue

        for _, player in roster.iterrows():
            pid = player["player_id"]
            player_history = latest[latest["player_id"] == pid]

            if player_history.empty:
                # No game log history — use season averages as fallback
                gp = max(player["games_played"], 1)
                row = {
                    "player_id": pid,
                    "name": f"{player['first_name']} {player['last_name']}",
                    "team": team,
                    "opponent": opponent,
                    "position": player["position"],
                    "is_home": int(is_home),
                    "rolling_goals_avg": player["goals"] / gp,
                    "rolling_shots_avg": player["shots"] / gp,
                    "rolling_points_avg": player["points"] / gp,
                    "rolling_toi_avg": 15.0,  # default estimate
                    "rolling_shooting_pct": player["shooting_pct"],
                    "games_in_window": min(gp, 10),
                    "is_forward": int(player["position"] in ["C", "L", "R"]),
                    "is_center": int(player["position"] == "C"),
                    "season_goals": player["goals"],
                    "season_gp": player["games_played"],
                }
            else:
                h = player_history.iloc[0]
                row = {
                    "player_id": pid,
                    "name": f"{player['first_name']} {player['last_name']}",
                    "team": team,
                    "opponent": opponent,
                    "position": player["position"],
                    "is_home": int(is_home),
                    "rolling_goals_avg": h.get("rolling_goals_avg", 0),
                    "rolling_shots_avg": h.get("rolling_shots_avg", 0),
                    "rolling_points_avg": h.get("rolling_points_avg", 0),
                    "rolling_toi_avg": h.get("rolling_toi_avg", 15.0),
                    "rolling_shooting_pct": h.get("rolling_shooting_pct", 0),
                    "games_in_window": h.get("games_in_window", 1),
                    "is_forward": int(player["position"] in ["C", "L", "R"]),
                    "is_center": int(player["position"] == "C"),
                    "season_goals": player["goals"],
                    "season_gp": player["games_played"],
                }

            all_predictions.append(row)

    return pd.DataFrame(all_predictions)


def predict_tonight() -> pd.DataFrame:
    """
    Generate goal scorer predictions for tonight's games.

    Returns:
        DataFrame with players ranked by goal-scoring probability.
    """
    print("🏒 SNIPE TRACKER — Tonight's Goal Scorer Predictions")
    print("=" * 55)

    # 1. Get tonight's schedule
    teams = _get_teams_playing_today()
    if not teams:
        print("No games scheduled today!")
        return pd.DataFrame()

    team_abbrevs = sorted(set(t["team"] for t in teams))
    print(f"\n📅 Teams playing: {', '.join(team_abbrevs)}")

    # 2. Load model
    print("🤖 Loading model...")
    model, scaler, meta = load_goal_model()
    print(f"   Model type: {meta['model_type']}")
    print(f"   Training AUC: {meta['metrics']['roc_auc']:.3f}")

    # 3. Build features
    print("⚙️  Building features...")
    game_log = load_game_data()
    pred_df = _build_prediction_features(teams, game_log)
    print(f"   {len(pred_df)} players across {len(team_abbrevs)} teams")

    # 4. Fill any NaN features with 0 (new players without history)
    pred_df[FEATURE_COLUMNS] = pred_df[FEATURE_COLUMNS].fillna(0)

    # 5. Predict
    print("🎯 Running predictions...\n")
    pred_df["goal_probability"] = predict_goal_probability(model, scaler, pred_df)
    pred_df = pred_df.sort_values("goal_probability", ascending=False)

    return pred_df


def print_top_picks(pred_df: pd.DataFrame, top_n: int = 25):
    """
    Pretty-print the top predicted goal scorers.
    """
    if pred_df.empty:
        print("No predictions available.")
        return

    print(f"\n🎯 TOP {top_n} MOST LIKELY GOAL SCORERS TONIGHT")
    print("=" * 70)

    display = pred_df.head(top_n).copy()
    display["prob_%"] = (display["goal_probability"] * 100).round(1)
    display["gpg"] = (display["season_goals"] / display["season_gp"].clip(lower=1)).round(2)
    display["matchup"] = display.apply(
        lambda r: f"{'vs' if r['is_home'] else '@'} {r['opponent']}", axis=1
    )

    cols = ["name", "team", "position", "matchup", "prob_%", "gpg",
            "rolling_goals_avg", "rolling_shots_avg", "season_goals"]
    headers = ["Player", "Team", "Pos", "Matchup", "Goal%", "GPG",
               "Roll G/Gm", "Roll S/Gm", "Season G"]

    print(tabulate(
        display[cols].values,
        headers=headers,
        tablefmt="simple",
        floatfmt=(".0f", ".0f", ".0f", ".0f", ".1f", ".2f", ".2f", ".1f", ".0f"),
    ))

    # Per-game breakdown
    print(f"\n\n📋 BREAKDOWN BY GAME")
    print("=" * 70)
    games = pred_df.groupby(["team", "opponent"]).head(5)
    for (team, opp), group in pred_df.groupby(["team", "opponent"]):
        top3 = group.head(3)
        matchup = f"{team} vs {opp}" if top3.iloc[0]["is_home"] else f"{team} @ {opp}"
        names = ", ".join(
            f"{r['name']} ({r['goal_probability']*100:.0f}%)"
            for _, r in top3.iterrows()
        )
        print(f"  {matchup}: {names}")


def run():
    """Main entry point for daily predictions."""
    pred_df = predict_tonight()
    if not pred_df.empty:
        print_top_picks(pred_df)
    return pred_df


if __name__ == "__main__":
    run()
