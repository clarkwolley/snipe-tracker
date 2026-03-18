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

    💡 CONCEPT: The scorecard shows how calibrated the model is.
    For each confidence tier (50-60%, 60-70%, etc), we check:
    - How many players we rated at that confidence
    - How many actually scored
    - Is the actual rate close to the predicted rate? (good calibration)
    
    Example: If we said "60% chance" for 10 players, we expect
    ~6 to score. If 6 do, we're well-calibrated ✅. 
    If only 2 score, we're overconfident 📉.
    """
    if graded.empty:
        print("No graded predictions to show.")
        return

    date = graded["prediction_date"].iloc[0]
    played = graded[graded["played"] == 1]

    print(f"\n{'='*70}")
    print(f"📋 SCORECARD — {date}")
    print(f"{'='*70}")

    # Simple overall stats
    total = len(played)
    actual_scorers = played["actual_scored"].sum()
    avg_prob = played["goal_probability"].mean() * 100
    brier = ((played["goal_probability"] - played["actual_scored"]) ** 2).mean()

    print(f"  Players tracked:    {total}")
    print(f"  Actually scored:    {actual_scorers}")
    print(f"  Avg confidence:     {avg_prob:.1f}%")
    print(f"  Brier score:        {brier:.4f}  (lower is better; 0=perfect)")

    # Calibration by tier — showing precision per tier
    print(f"\n  📊 CALIBRATION BY CONFIDENCE TIER:")
    print(f"  (For each tier, are actual scoring rates close to predicted?)\n")
    
    bins = [
        (0.55, 1.0, "🔥 HIGH   (55%+)"),
        (0.45, 0.55, "🎯 MID    (45-55%)"),
        (0.35, 0.45, "👀 LOW    (35-45%)"),
        (0.0, 0.35, "📋 MINIMAL (<35%)"),
    ]

    tier_results = []
    for lo, hi, label in bins:
        tier = played[(played["goal_probability"] >= lo) & (played["goal_probability"] < hi)]
        if tier.empty:
            continue
        
        tier_scored = tier["actual_scored"].sum()
        tier_total = len(tier)
        avg_pred_pct = tier["goal_probability"].mean() * 100
        actual_pct = (tier_scored / tier_total * 100) if tier_total > 0 else 0
        calibration_diff = actual_pct - avg_pred_pct
        
        # Emoji for calibration: ✅ if within ±8%, 📈 if underestimating, 📉 if overestimating
        cal_emoji = "✅" if abs(calibration_diff) <= 8 else ("📈" if calibration_diff > 8 else "📉")
        
        tier_results.append({
            "label": label,
            "tier": tier,
            "scored": tier_scored,
            "total": tier_total,
            "avg_pred": avg_pred_pct,
            "actual": actual_pct,
            "cal_emoji": cal_emoji,
        })
    
    for res in tier_results:
        print(f"    {res['label']}")
        print(f"      Players:  {res['scored']}/{res['total']} scored ({res['actual']:.0f}%)")
        print(f"      Expected: ~{res['avg_pred']:.0f}%  →  Actual: {res['actual']:.0f}%  {res['cal_emoji']}")
        print()

    # Top scorers by confidence
    print(f"  🏆 TOP SCORERS (ranked by our confidence):")
    top_by_prob = played.nlargest(5, "goal_probability")
    for rank, (_, row) in enumerate(top_by_prob.iterrows(), 1):
        status = "✅" if row["actual_scored"] == 1 else "❌"
        goals = int(row["actual_goals"])
        shots = int(row["actual_shots"])
        print(f"    {rank}. {status} {row['name']:25s} {row['goal_probability']*100:5.1f}% → {goals}G on {shots}S")

    # Surprise scorers (high actual but low predicted) and duds (high pred, low actual)
    print(f"\n  💥 SURPRISE SCORERS (low confidence, actually scored):")
    surprise = played[
        (played["goal_probability"] < 0.35) & (played["actual_scored"] == 1)
    ].nlargest(5, "actual_goals")
    if not surprise.empty:
        for _, row in surprise.iterrows():
            print(f"    🎁 {row['name']:25s} {row['goal_probability']*100:5.1f}% predicted → {int(row['actual_goals'])}G")
    else:
        print(f"    (none)")

    print(f"\n  💔 HIGH CONFIDENCE MISSES (55%+, didn't score):")
    duds = played[
        (played["goal_probability"] >= 0.55) & (played["actual_scored"] == 0)
    ].nlargest(5, "goal_probability")
    if not duds.empty:
        for _, row in duds.iterrows():
            print(f"    ❌ {row['name']:25s} {row['goal_probability']*100:5.1f}% predicted → {int(row['actual_shots'])} shots, no goal")
    else:
        print(f"    (none)")

    print(f"\n{'='*70}")


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
    actual_scorers = played["actual_scored"].sum()
    avg_prob = played["goal_probability"].mean() * 100

    print(f"\n{'='*70}")
    print(f"📈 LIFETIME MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"  Days tracked:     {dates}")
    print(f"  Total picks:      {total}")
    print(f"  Actual scorers:   {actual_scorers} ({actual_scorers/total*100:.1f}%)")
    print(f"  Avg confidence:   {avg_prob:.1f}%")

    # Brier score (best metric for probability predictions)
    brier = ((played["goal_probability"] - played["actual_scored"]) ** 2).mean()
    print(f"  Brier score:      {brier:.4f} (lower is better; 0=perfect)")

    # Calibration by tier across all dates
    print(f"\n  📊 CALIBRATION ACROSS ALL DAYS:")
    print(f"  (Averaging calibration across {dates} days)\n")
    
    bins = [
        (0.55, 1.0, "🔥 HIGH   (55%+)"),
        (0.45, 0.55, "🎯 MID    (45-55%)"),
        (0.35, 0.45, "👀 LOW    (35-45%)"),
        (0.0, 0.35, "📋 MINIMAL (<35%)"),
    ]

    for lo, hi, label in bins:
        tier = played[(played["goal_probability"] >= lo) & (played["goal_probability"] < hi)]
        if tier.empty:
            continue
        
        avg_pred = tier["goal_probability"].mean() * 100
        actual_rate = tier["actual_scored"].mean() * 100
        n = len(tier)
        diff = actual_rate - avg_pred
        
        # Color code: well-calibrated if within ±8%
        cal_emoji = "✅" if abs(diff) <= 8 else ("📈" if diff > 8 else "📉")
        
        print(f"    {label}")
        print(f"      {n:4d} picks → Expected ~{avg_pred:.0f}% | Actual {actual_rate:.0f}% {cal_emoji}")

    print(f"\n{'='*70}")
