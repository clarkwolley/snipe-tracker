"""
Prediction tracker — saves picks and grades them against actual results.

💡 KEY CONCEPT: A model that isn't tracked is a model you can't improve.
This module does three things:
1. Saves each day's predictions to a ledger (CSV)
2. After games finish, pulls actual results and grades each pick
3. Calculates running accuracy metrics over time

This is called "backtesting" in the betting/finance world — the
honest scorecard of whether your model actually works.
"""

import os
from datetime import datetime

import pandas as pd
import numpy as np

from src.data import nhl_api
from src.data.collector import get_game_player_stats


TRACKER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PICKS_FILE = os.path.join(TRACKER_DIR, "picks_ledger.csv")
GRADED_FILE = os.path.join(TRACKER_DIR, "graded_ledger.csv")


def save_predictions(pred_df: pd.DataFrame):
    """
    Save today's predictions to the running ledger.

    Each row = one player prediction for one date.
    We append to the file so it grows over time.

    💡 CONCEPT: The ledger is your "paper trail." You can always
    go back and see what the model predicted on any given date.
    """
    # Columns to persist — include new features for historical analysis
    save_cols = [
        "player_id", "name", "team", "opponent", "position",
        "is_home", "goal_probability", "rolling_goals_avg",
        "rolling_shots_avg", "season_goals", "season_gp",
        # New features
        "goal_streak", "point_streak", "drought", "is_hot",
        "shots_per_toi", "high_volume_shooter",
        "opp_goalie_save_pct", "opp_goalie_gaa", "opp_goalie_name",
        "is_back_to_back", "days_rest",
    ]
    # Only save columns that actually exist (backward compat)
    available_cols = [c for c in save_cols if c in pred_df.columns]
    picks = pred_df[available_cols].copy()
    picks["prediction_date"] = datetime.now().strftime("%Y-%m-%d")
    picks["predicted_at"] = datetime.now().isoformat()

    if os.path.exists(PICKS_FILE):
        existing = pd.read_csv(PICKS_FILE)
        # Remove any existing picks for today (re-run safe)
        existing = existing[existing["prediction_date"] != picks["prediction_date"].iloc[0]]
        combined = pd.concat([existing, picks], ignore_index=True)
    else:
        combined = picks

    combined.to_csv(PICKS_FILE, index=False)
    print(f"💾 Saved {len(picks)} predictions to ledger ({len(combined)} total rows)")


def grade_predictions(date_str: str) -> pd.DataFrame:
    """
    Grade predictions for a specific date against actual results.

    Pulls boxscores for all games on that date, checks which
    predicted scorers actually scored, and calculates accuracy.

    Args:
        date_str: Date to grade in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with predictions + actual results.

    💡 CONCEPT: "Grading" means comparing what we predicted to
    what actually happened. For each player we ask:
    - Did they play? (injuries/scratches happen)
    - Did they score? (the actual outcome)
    - How confident were we? (the predicted probability)
    """
    if not os.path.exists(PICKS_FILE):
        print("No predictions to grade! Run predictions first.")
        return pd.DataFrame()

    ledger = pd.read_csv(PICKS_FILE)
    day_picks = ledger[ledger["prediction_date"] == date_str].copy()

    if day_picks.empty:
        print(f"No predictions found for {date_str}")
        return pd.DataFrame()

    # Pull actual game results for that date
    print(f"📊 Fetching actual results for {date_str}...")
    schedule = nhl_api.get_schedule(date_str)
    game_ids = []
    for day in schedule.get("gameWeek", []):
        if day["date"] != date_str:
            continue
        for game in day.get("games", []):
            if game["gameState"] == "OFF":
                game_ids.append(game["id"])

    if not game_ids:
        print(f"No completed games found for {date_str} (games may not have started yet)")
        return pd.DataFrame()

    # Collect actual player stats
    actual_frames = []
    for gid in game_ids:
        try:
            gdf = get_game_player_stats(gid)
            actual_frames.append(gdf)
        except Exception as e:
            print(f"  ⚠️  Failed to fetch game {gid}: {e}")

    if not actual_frames:
        return pd.DataFrame()

    actuals = pd.concat(actual_frames, ignore_index=True)
    actuals["actual_goals"] = actuals["goals"]
    actuals["actual_scored"] = (actuals["goals"] > 0).astype(int)
    actuals["actual_shots"] = actuals["shots"]

    # Merge predictions with actuals
    graded = day_picks.merge(
        actuals[["player_id", "actual_goals", "actual_scored", "actual_shots"]],
        on="player_id",
        how="left",
    )

    # Players who didn't play (scratched/injured)
    graded["played"] = graded["actual_goals"].notna().astype(int)
    graded["actual_goals"] = graded["actual_goals"].fillna(0).astype(int)
    graded["actual_scored"] = graded["actual_scored"].fillna(0).astype(int)
    graded["actual_shots"] = graded["actual_shots"].fillna(0).astype(int)

    # Was our prediction correct?
    graded["predicted_goal"] = (graded["goal_probability"] >= 0.65).astype(int)
    graded["correct"] = (graded["predicted_goal"] == graded["actual_scored"]).astype(int)
    graded["hit"] = ((graded["predicted_goal"] == 1) & (graded["actual_scored"] == 1)).astype(int)

    return graded


def save_graded(graded: pd.DataFrame):
    """Append graded results to the graded ledger."""
    if graded.empty:
        return

    if os.path.exists(GRADED_FILE):
        existing = pd.read_csv(GRADED_FILE)
        date = graded["prediction_date"].iloc[0]
        existing = existing[existing["prediction_date"] != date]
        combined = pd.concat([existing, graded], ignore_index=True)
    else:
        combined = graded

    combined.to_csv(GRADED_FILE, index=False)
    print(f"💾 Saved graded results ({len(combined)} total rows)")


def print_scorecard(graded: pd.DataFrame):
    """
    Pretty-print the grading results for a single day.

    💡 CONCEPT: The scorecard shows you how calibrated the model is.
    If the model says "70% chance" for 10 players, roughly 7 should
    actually score. If only 2 do, the model is overconfident.
    """
    if graded.empty:
        print("No graded predictions to show.")
        return

    date = graded["prediction_date"].iloc[0]
    played = graded[graded["played"] == 1]

    print(f"\n{'='*65}")
    print(f"📋 SCORECARD — {date}")
    print(f"{'='*65}")

    # Overall stats
    total = len(played)
    actual_scorers = played["actual_scored"].sum()
    predicted_scorers = played["predicted_goal"].sum()
    hits = played["hit"].sum()
    correct = played["correct"].sum()

    print(f"  Players tracked:     {total}")
    print(f"  Actually scored:     {actual_scorers}")
    print(f"  We predicted goal:   {predicted_scorers}")
    print(f"  Correct predictions: {correct}/{total} ({correct/max(total,1)*100:.1f}%)")
    print(f"  Hits (predicted + scored): {hits}/{predicted_scorers} "
          f"({hits/max(predicted_scorers,1)*100:.1f}% precision)")

    # Calibration by tier
    print(f"\n  📊 CALIBRATION BY CONFIDENCE TIER:")
    bins = [(0.72, 1.0, "🔥 FIRE  (72%+)"),
            (0.68, 0.72, "🎯 STRONG (68-72%)"),
            (0.64, 0.68, "👀 WATCH  (64-68%)"),
            (0.0, 0.64, "📋 OTHER  (<64%)")]

    for lo, hi, label in bins:
        tier = played[(played["goal_probability"] >= lo) & (played["goal_probability"] < hi)]
        if tier.empty:
            continue
        tier_scored = tier["actual_scored"].sum()
        tier_total = len(tier)
        avg_prob = tier["goal_probability"].mean() * 100
        actual_rate = tier_scored / tier_total * 100
        print(f"    {label}: {tier_scored}/{tier_total} scored "
              f"(predicted ~{avg_prob:.0f}%, actual {actual_rate:.0f}%)")

    # Top picks that hit
    print(f"\n  ✅ TOP PICKS THAT HIT:")
    top_hits = played[(played["actual_scored"] == 1)].nlargest(10, "goal_probability")
    for _, row in top_hits.iterrows():
        print(f"    ✅ {row['name']} ({row['team']}) — "
              f"{row['goal_probability']*100:.0f}% predicted, "
              f"{row['actual_goals']} goal(s), {row['actual_shots']} shots")

    # Top picks that missed
    print(f"\n  ❌ TOP PICKS THAT MISSED:")
    top_misses = played[
        (played["predicted_goal"] == 1) & (played["actual_scored"] == 0)
    ].nlargest(5, "goal_probability")
    for _, row in top_misses.iterrows():
        print(f"    ❌ {row['name']} ({row['team']}) — "
              f"{row['goal_probability']*100:.0f}% predicted, "
              f"{row['actual_shots']} shots but no goal")

    print(f"{'='*65}")


def run_grading(date_str: str):
    """Full grading pipeline for a specific date."""
    graded = grade_predictions(date_str)
    if not graded.empty:
        print_scorecard(graded)
        save_graded(graded)
    return graded


def lifetime_stats():
    """
    Print running stats across all graded predictions.

    💡 CONCEPT: One day's results are noisy — you could get
    lucky or unlucky. The lifetime stats smooth that out
    and show the model's TRUE performance over time.
    """
    if not os.path.exists(GRADED_FILE):
        print("No graded predictions yet! Grade some days first.")
        return

    df = pd.read_csv(GRADED_FILE)
    played = df[df["played"] == 1]

    dates = played["prediction_date"].nunique()
    total = len(played)

    print(f"\n{'='*65}")
    print(f"📈 LIFETIME MODEL PERFORMANCE")
    print(f"{'='*65}")
    print(f"  Days tracked:    {dates}")
    print(f"  Total picks:     {total}")
    print(f"  Actual scorers:  {played['actual_scored'].sum()}")

    predicted = played[played["predicted_goal"] == 1]
    if len(predicted) > 0:
        precision = predicted["actual_scored"].mean() * 100
        print(f"  Predicted goals: {len(predicted)}")
        print(f"  Hit rate:        {precision:.1f}% (of 'goal' predictions that scored)")

    # Overall accuracy
    accuracy = played["correct"].mean() * 100
    print(f"  Accuracy:        {accuracy:.1f}%")

    # Brier score over time
    brier = ((played["goal_probability"] - played["actual_scored"]) ** 2).mean()
    print(f"  Brier score:     {brier:.3f} (lower is better)")

    # Calibration
    print(f"\n  📊 OVERALL CALIBRATION:")
    for lo, hi, label in [
        (0.72, 1.0, "🔥 FIRE  (72%+)"),
        (0.68, 0.72, "🎯 STRONG (68-72%)"),
        (0.64, 0.68, "👀 WATCH  (64-68%)"),
        (0.0, 0.64, "📋 OTHER  (<64%)"),
    ]:
        tier = played[(played["goal_probability"] >= lo) & (played["goal_probability"] < hi)]
        if tier.empty:
            continue
        avg_pred = tier["goal_probability"].mean() * 100
        actual_rate = tier["actual_scored"].mean() * 100
        n = len(tier)
        diff = actual_rate - avg_pred
        arrow = "✅" if abs(diff) < 10 else ("📈" if diff > 0 else "📉")
        print(f"    {label}: predicted ~{avg_pred:.0f}% | actual {actual_rate:.0f}% "
              f"| n={n} {arrow}")

    print(f"{'='*65}")
