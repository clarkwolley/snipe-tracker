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
from src.data.collector import (
    get_todays_games,
    get_team_skaters,
    get_back_to_back_status,
)
from src.data.history import load_game_data, refresh_game_log
from src.features.player_features import (
    build_player_features,
    FEATURE_COLUMNS,
)
from src.features.goalie_features import build_goalie_matchup_features
from src.models.goal_model import load_goal_model, predict_goal_probability
from src.data.injuries import get_unavailable_players, print_injury_report
from src.config import PDO_SELL_HIGH_THRESHOLD, SELL_HIGH_SCORING_PACE


def _get_teams_playing_today() -> list[dict]:
    """
    Get all teams playing in today's scheduled games.

    Returns:
        List of dicts with team abbreviation and home/away status.
    """
    schedule = get_todays_games()
    if schedule.empty:
        return []

    game_date = schedule["date"].iloc[0]
    teams = []
    for _, game in schedule.iterrows():
        teams.append({"team": game["home_team"], "is_home": True,
                      "opponent": game["away_team"], "game_id": game["game_id"],
                      "game_date": game_date})
        teams.append({"team": game["away_team"], "is_home": False,
                      "opponent": game["home_team"], "game_id": game["game_id"],
                      "game_date": game_date})
    return teams


def _build_prediction_features(teams: list[dict], game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features for all players on tonight's teams.

    Strategy:
    1. Get season stats for each team's roster (current ability)
    2. Look up each player's recent game log (recent form + streaks)
    3. Inject goalie matchup features (opposing goalie quality)
    4. Inject back-to-back / rest features
    5. Combine into the features our model expects

    💡 CONCEPT: For predictions, we need the same features the model
    was trained on. The rolling averages come from the player's
    RECENT games — the last 10 games they played.
    """
    # Get recent game log for rolling averages + streaks
    featured = build_player_features(game_log)

    # For each player, get their LATEST rolling features
    latest = (
        featured
        .sort_values("game_date")
        .groupby("player_id")
        .last()
        .reset_index()
    )

    # Pre-fetch goalie matchup data per opponent (avoid duplicate API calls)
    opponents = set(t["opponent"] for t in teams)
    goalie_cache = {}
    for opp in opponents:
        try:
            goalie_cache[opp] = build_goalie_matchup_features(opp)
        except Exception as e:
            print(f"  ⚠️  Could not fetch goalie for {opp}: {e}")
            goalie_cache[opp] = {
                "opp_goalie_save_pct": 0.900, "opp_goalie_gaa": 3.00,
                "opp_goalie_quality": 7.5, "opp_goalie_name": "Unknown",
            }

    # Pre-fetch back-to-back status per team
    b2b_cache = {}
    for t in teams:
        team = t["team"]
        game_date = t.get("game_date", "")
        if team not in b2b_cache and game_date:
            try:
                b2b_cache[team] = get_back_to_back_status(team, game_date)
            except Exception:
                b2b_cache[team] = {"is_back_to_back": False, "days_rest": 2}

    # Pre-fetch own-team goalie save % for PDO calculation.
    # PDO = player SH% + team SV%.  High PDO (>103) signals luck
    # that will regress, making the player a sell-high candidate.
    from src.features.goalie_features import get_likely_starter
    team_sv_cache = {}
    for t in teams:
        team = t["team"]
        if team not in team_sv_cache:
            try:
                starter = get_likely_starter(team)
                team_sv_cache[team] = starter["goalie_save_pct"]
            except Exception:
                team_sv_cache[team] = 0.900

    all_predictions = []

    for team_info in teams:
        team = team_info["team"]
        is_home = team_info["is_home"]
        opponent = team_info["opponent"]
        goalie_feats = goalie_cache.get(opponent, {})
        rest_info = b2b_cache.get(team, {"is_back_to_back": False, "days_rest": 2})

        try:
            roster = get_team_skaters(team)
        except Exception as e:
            print(f"  ⚠️  Could not fetch roster for {team}: {e}")
            print(f"       → ALL players on {team} excluded from predictions!")
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
                    "rolling_toi_avg": 15.0,
                    "rolling_shooting_pct": player["shooting_pct"],
                    "games_in_window": min(gp, 10),
                    # Streak features — no history, default to 0
                    "goal_streak": 0,
                    "point_streak": 0,
                    "drought": 0,
                    "is_hot": 0,
                    # Shot quality — default estimates
                    "shots_per_toi": player["shots"] / gp / 15.0,
                    "shooting_pct_trend": 0.0,
                    "high_volume_shooter": int(player["shots"] / gp > 3.0),
                    # PP production
                    "rolling_pp_goals_avg": player.get("pp_goals", 0) / gp,
                    # Player attributes
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
                    # Streak features from game log
                    "goal_streak": int(h.get("goal_streak", 0)),
                    "point_streak": int(h.get("point_streak", 0)),
                    "drought": int(h.get("drought", 0)),
                    "is_hot": int(h.get("is_hot", 0)),
                    # Shot quality from game log
                    "shots_per_toi": h.get("shots_per_toi", 0.0),
                    "shooting_pct_trend": h.get("shooting_pct_trend", 0.0),
                    "high_volume_shooter": int(h.get("high_volume_shooter", 0)),
                    # PP production
                    "rolling_pp_goals_avg": h.get("rolling_pp_goals_avg", 0),
                    # Player attributes
                    "is_forward": int(player["position"] in ["C", "L", "R"]),
                    "is_center": int(player["position"] == "C"),
                    "season_goals": player["goals"],
                    "season_gp": player["games_played"],
                }

            # Inject goalie matchup features
            row["opp_goalie_save_pct"] = goalie_feats.get("opp_goalie_save_pct", 0.900)
            row["opp_goalie_gaa"] = goalie_feats.get("opp_goalie_gaa", 3.00)
            row["opp_goalie_quality"] = goalie_feats.get("opp_goalie_quality", 7.5)
            row["opp_goalie_name"] = goalie_feats.get("opp_goalie_name", "Unknown")

            # Inject rest features
            row["is_back_to_back"] = int(rest_info.get("is_back_to_back", False))
            row["days_rest"] = rest_info.get("days_rest", 2)

            # Own-team goalie SV% for PDO calculation
            row["team_sv_pct"] = team_sv_cache.get(team, 0.900)

            all_predictions.append(row)

    return pd.DataFrame(all_predictions)


def _log_leader_coverage(pred_df: pd.DataFrame, teams: list[dict]) -> None:
    """
    Log where the league's top goal scorers rank in tonight's predictions.

    This makes it immediately obvious whether a league leader:
    - Isn't playing tonight (team not scheduled)
    - Was excluded by injury filtering
    - Is playing but ranked lower than expected

    Helps debug "why isn't Player X in the top picks?" questions.
    """
    playing_teams = {t["team"] for t in teams}

    # Rank by season goals to approximate "league leaders"
    if pred_df.empty or "season_goals" not in pred_df.columns:
        return

    top_scorers = pred_df.nlargest(10, "season_goals")
    print("\n📊 LEAGUE LEADER CHECK (top scorers playing tonight):")

    for i, (_, row) in enumerate(top_scorers.iterrows(), 1):
        rank_in_preds = pred_df.index.get_loc(row.name) + 1
        prob_pct = row["goal_probability"] * 100
        note = ""
        if row.get("sell_high", 0):
            note += " [📉 SELL HIGH]"
        if row.get("is_back_to_back", 0):
            note += " [B2B]"
        if row.get("injury_note", "") == "DTD":
            note += " [DTD]"

        print(
            f"   {i:2d}. {row['name']:25s} ({row['team']}) "
            f"— {int(row['season_goals'])}G this season → "
            f"ranked #{rank_in_preds} tonight ({prob_pct:.1f}%){note}"
        )


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
    print(f"\n📅 Teams playing ({len(team_abbrevs)} teams): {', '.join(team_abbrevs)}")
    print(f"   ℹ️  Only players on tonight's teams can appear in predictions.")

    # Show back-to-back warnings
    for t in teams:
        game_date = t.get("game_date", "")
        if game_date:
            try:
                b2b = get_back_to_back_status(t["team"], game_date)
                if b2b["is_back_to_back"]:
                    print(f"   ⚠️  {t['team']} is on a BACK-TO-BACK (fatigue factor!)")
            except Exception:
                pass

    # 1b. Fetch injury report
    print("\n🏥 Checking injury report...")
    injury_df = print_injury_report(teams=team_abbrevs)
    unavailable = get_unavailable_players(teams=team_abbrevs)
    if unavailable:
        print(f"\n   → {len(unavailable)} players will be excluded from predictions")

    # 2. Load model
    print("🤖 Loading model...")
    model, scaler, meta = load_goal_model()
    print(f"   Model type: {meta['model_type']}")
    print(f"   Training AUC: {meta['metrics']['roc_auc']:.3f}")

    # 3. Build features (refresh game log first — stale data = bad streaks)
    print("⚙️  Refreshing game log...")
    game_log = refresh_game_log()
    pred_df = _build_prediction_features(teams, game_log)
    print(f"   {len(pred_df)} players across {len(team_abbrevs)} teams")

    # 4. Filter out injured players (Out / IR / Suspended)
    if unavailable:
        before = len(pred_df)
        pred_df = pred_df[~pred_df["name"].isin(unavailable)].reset_index(drop=True)
        excluded = before - len(pred_df)
        if excluded:
            print(f"   🚫 Excluded {excluded} injured/suspended players")

    # Annotate Day-to-Day players so they show in output
    if not injury_df.empty:
        dtd_players = set(
            injury_df[injury_df["status"] == "Day-To-Day"]["player_name"]
        )
        pred_df["injury_note"] = pred_df["name"].apply(
            lambda n: "DTD" if n in dtd_players else ""
        )
    else:
        pred_df["injury_note"] = ""

    # 5. Fill any NaN features with 0 (new players without history)
    pred_df[FEATURE_COLUMNS] = pred_df[FEATURE_COLUMNS].fillna(0)

    # 6. Predict
    print("🎯 Running predictions...\n")
    pred_df["goal_probability"] = predict_goal_probability(
        model, scaler, pred_df, meta=meta
    )
    pred_df = pred_df.sort_values("goal_probability", ascending=False)

    # 7. PDO regression detection
    # PDO = (personal SH% * 100) + (team SV% * 100)
    # League-average PDO ~ 100.  Values above 103 are unsustainable
    # and predict regression — a "sell high" signal.
    pred_df["pdo"] = (
        pred_df["rolling_shooting_pct"] * 100
        + pred_df.get("team_sv_pct", pd.Series([90.0] * len(pred_df))).fillna(90.0) * 100
    )
    pred_df["sell_high"] = (
        (pred_df["rolling_goals_avg"] >= SELL_HIGH_SCORING_PACE)
        & (pred_df["pdo"] > PDO_SELL_HIGH_THRESHOLD)
    ).astype(int)

    # Sanity check: surface league leaders so we can tell at a glance
    # whether top scorers are playing tonight and where they rank.
    _log_leader_coverage(pred_df, teams)

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
    # Streak indicator: 🔥 for hot, ❄️ for drought 5+, blank otherwise
    display["streak"] = display.apply(
        lambda r: f"🔥{int(r.get('goal_streak', 0))}" if r.get("is_hot", 0)
        else (f"❄️{int(r.get('drought', 0))}" if r.get("drought", 0) >= 5 else ""),
        axis=1,
    )
    # Sell-high indicator: 📉 for PDO regression candidates
    if "sell_high" in display.columns:
        display["streak"] = display.apply(
            lambda r: f"📉SH {r['streak']}" if r.get("sell_high", 0)
            else r["streak"],
            axis=1,
        )
    # Injury annotation: ⚠️ DTD for day-to-day players
    if "injury_note" in display.columns:
        display["streak"] = display.apply(
            lambda r: f"⚠️DTD {r['streak']}" if r.get("injury_note") == "DTD"
            else r["streak"],
            axis=1,
        )
    # Goalie info
    display["vs_goalie"] = display.get("opp_goalie_name", "")

    cols = ["name", "team", "position", "matchup", "prob_%", "streak",
            "gpg", "rolling_shots_avg", "season_goals", "vs_goalie"]
    headers = ["Player", "Team", "Pos", "Matchup", "Goal%", "Streak",
               "GPG", "Roll S/Gm", "Season G", "vs Goalie"]

    print(tabulate(
        display[cols].values,
        headers=headers,
        tablefmt="simple",
        floatfmt=(".0f", ".0f", ".0f", ".0f", ".1f", ".0f", ".2f", ".1f", ".0f", ".0f"),
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


def predict_game_winners() -> pd.DataFrame:
    """
    Predict winners for tonight's games.

    Returns:
        DataFrame with one row per game, home win probability.
    """
    from src.models.game_model import load_game_model, predict_game_winner
    from src.data.collector import get_standings_df
    from src.features.team_features import build_team_strength

    schedule = get_todays_games()
    if schedule.empty:
        return pd.DataFrame()

    try:
        model, scaler, meta = load_game_model()
    except FileNotFoundError:
        print("⚠️  No game winner model trained yet. Skipping.")
        return pd.DataFrame()

    standings = get_standings_df()
    strength = build_team_strength(standings)

    rows = []
    for _, game in schedule.iterrows():
        prob = predict_game_winner(game["home_team"], game["away_team"], model, scaler)
        winner = game["home_team"] if prob > 0.5 else game["away_team"]
        confidence = max(prob, 1 - prob) * 100

        home_strength = strength[strength["team"] == game["home_team"]]
        away_strength = strength[strength["team"] == game["away_team"]]

        rows.append({
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win_prob": round(prob * 100, 1),
            "away_win_prob": round((1 - prob) * 100, 1),
            "predicted_winner": winner,
            "confidence": round(confidence, 1),
            "home_pts": home_strength["point_pct"].values[0] if len(home_strength) else 0,
            "away_pts": away_strength["point_pct"].values[0] if len(away_strength) else 0,
            "home_pp_pct": home_strength["pp_pct"].values[0] if len(home_strength) else 0.20,
            "away_pp_pct": away_strength["pp_pct"].values[0] if len(away_strength) else 0.20,
        })

    return pd.DataFrame(rows)


def print_game_picks(game_df: pd.DataFrame):
    """Pretty-print game winner predictions."""
    if game_df.empty:
        return

    print(f"\n\n🏆 GAME WINNER PREDICTIONS")
    print("=" * 70)

    for _, g in game_df.iterrows():
        winner = g["predicted_winner"]
        is_home_fav = g["home_win_prob"] > 50
        emoji = "🏠" if is_home_fav else "✈️"
        conf = g["confidence"]
        conf_bar = "🟢" if conf >= 60 else "🟡" if conf >= 55 else "⚪"

        print(f"  {g['away_team']} ({g['away_pts']:.3f}) @ {g['home_team']} ({g['home_pts']:.3f})")
        print(f"    → {conf_bar} {emoji} {winner} wins ({conf:.1f}% confidence)")
        print(f"      Home: {g['home_win_prob']}% | Away: {g['away_win_prob']}%")
        print()


def run():
    """Main entry point for daily predictions."""
    pred_df = predict_tonight()
    if not pred_df.empty:
        print_top_picks(pred_df)

    # Game winner predictions
    game_df = predict_game_winners()
    if not game_df.empty:
        print_game_picks(game_df)

    if not pred_df.empty:
        # Save predictions to tracker ledger
        from src.predictions.tracker import save_predictions
        save_predictions(pred_df)

        # Generate shareable HTML report
        from src.predictions.report import generate_html_report
        report_path = generate_html_report(pred_df, game_df=game_df)
        print(f"\n🌐 Open in browser: file://{report_path}")

    return pred_df


if __name__ == "__main__":
    run()
