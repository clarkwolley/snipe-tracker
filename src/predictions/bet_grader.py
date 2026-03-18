"""
Bet grader — reconcile your bets with actual game results.

💡 KEY CONCEPT: Your bets are on "Anytime Scorer" (player scores at least 1 goal).
This module:
1. Loads your open bets for a date
2. Loads the graded predictions for that date
3. Matches players and grades the bets
4. Shows ROI and calibration metrics
"""

import pandas as pd
from pathlib import Path

from src.predictions.bet_tracker import BetTracker
from src.predictions.tracker import grade_predictions


def grade_bets_for_date(game_date: str) -> dict:
    """
    Grade all open bets for a given game date.
    
    Args:
        game_date: Date in YYYY-MM-DD format
    
    Returns:
        Dict with graded bets and summary stats
    """
    # Load bets
    tracker = BetTracker()
    open_bets = tracker.list_open(game_date=game_date)
    
    if not open_bets:
        return {
            "date": game_date,
            "bets": [],
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "roi": 0.0,
        }
    
    # Load graded predictions
    graded = grade_predictions(game_date)
    
    if graded.empty:
        print(f"⚠️  No graded predictions found for {game_date}. Can't grade bets.")
        return {
            "date": game_date,
            "bets": open_bets,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "roi": 0.0,
            "error": "No predictions graded for this date",
        }
    
    # Match and grade
    graded_bets = []
    wins = 0
    losses = 0
    total_profit = 0.0
    
    for bet in open_bets:
        player_name = bet["player_name"]
        
        # Find matching prediction
        matches = graded[
            graded["name"].str.contains(player_name, case=False, na=False)
        ]
        
        if matches.empty:
            # Player not in graded predictions (didn't play?)
            graded_bets.append({
                **bet,
                "actual_goals": "N/A (didn't play)",
                "status": "cancelled",
                "profit": "0.00",
            })
            continue
        
        # Take best match (if multiple)
        match = matches.iloc[0]
        actual_goals = int(match["actual_goals"])
        
        # Grade the bet (anytime scorer = 1+ goals)
        won = actual_goals >= 1
        bet_amount = float(bet["bet_amount"])
        odds = float(bet["odds"])
        
        # Calculate payout
        if odds < 0:
            payout = bet_amount * (100 / abs(odds)) if won else 0.0
        else:
            payout = bet_amount * (odds / 100) if won else 0.0
        
        profit = payout - bet_amount
        
        if won:
            wins += 1
        else:
            losses += 1
        
        total_profit += profit
        
        graded_bets.append({
            **bet,
            "actual_goals": str(actual_goals),
            "status": "won" if won else "lost",
            "profit": f"{profit:.2f}",
        })
        
        # Update tracker
        tracker.grade_bet(int(bet["bet_id"]), actual_goals)
    
    total_wagered = sum(float(b["bet_amount"]) for b in open_bets)
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    
    return {
        "date": game_date,
        "bets": graded_bets,
        "wins": wins,
        "losses": losses,
        "total_wagered": total_wagered,
        "profit": total_profit,
        "roi": roi,
    }


def print_bet_scorecard(game_date: str):
    """
    Pretty-print betting results for a date.
    """
    results = grade_bets_for_date(game_date)
    
    print(f"\n{'='*70}")
    print(f"💰 BETTING SCORECARD — {game_date}")
    print(f"{'='*70}")
    
    if "error" in results:
        print(f"  ⚠️  {results['error']}")
        return
    
    if not results["bets"]:
        print(f"  No bets for this date")
        return
    
    # Summary
    print(f"\n  Record:    {results['wins']}-{results['losses']}")
    if results["wins"] + results["losses"] > 0:
        win_pct = results["wins"] / (results["wins"] + results["losses"]) * 100
        print(f"  Win %:     {win_pct:.1f}%")
    print(f"  Wagered:   ${results['total_wagered']:.2f}")
    print(f"  Profit:    ${results['profit']:+.2f}")
    print(f"  ROI:       {results['roi']:+.1f}%")
    
    # Detailed results
    print(f"\n  📋 BETS:")
    print(f"  {'#':>2} {'Player':<20} {'Amount':>7} {'Odds':>6} {'Result':>8} {'Profit':>8}")
    print(f"  {'-'*60}")
    
    for bet in results["bets"]:
        name = bet["player_name"][:20]
        amount = float(bet["bet_amount"])
        odds = bet["odds"]
        status = bet["status"]
        profit = float(bet["profit"]) if bet["profit"] != "N/A" else 0
        
        emoji = "✅" if status == "won" else ("❌" if status == "lost" else "⚪")
        
        print(
            f"  {bet['bet_id']:>2} {emoji} {name:<20} ${amount:>6.2f} {odds:>6} "
            f"{status:>8} ${profit:>7.2f}"
        )
    
    print(f"\n{'='*70}")


def compare_predictions_vs_bets(game_date: str):
    """
    Compare your model predictions vs. your actual bets.
    
    Useful for: Did you bet on the high-confidence picks?
    Or did you go off-script?
    """
    from src.predictions.tracker import grade_predictions
    
    # Load graded predictions
    graded = grade_predictions(game_date)
    
    if graded.empty:
        print(f"No predictions for {game_date}")
        return
    
    # Load your bets
    tracker = BetTracker()
    bets = tracker.list_all(game_date=game_date)
    
    if not bets:
        print(f"No bets for {game_date}")
        return
    
    # Match them
    played = graded[graded["played"] == 1]
    
    print(f"\n{'='*70}")
    print(f"🤔 PREDICTIONS vs. BETS — {game_date}")
    print(f"{'='*70}")
    print(f"\n  Your bets vs. model's top picks:\n")
    
    # Top predictions
    top_preds = played.nlargest(5, "goal_probability")
    
    bet_names = {b["player_name"].lower() for b in bets}
    
    for i, (_, pred) in enumerate(top_preds.iterrows(), 1):
        name = pred["name"]
        prob = pred["goal_probability"]
        scored = pred["actual_scored"]
        
        bet_this = any(name.lower() in bn.lower() or bn.lower() in name.lower() 
                       for bn in bet_names)
        
        emoji = "🎯" if bet_this else "  "
        scored_emoji = "✅" if scored else "❌"
        
        print(f"    {i}. {emoji} {name:<25} {prob*100:5.1f}% → {scored_emoji}")
    
    print(f"\n  🎯 = You bet on this pick")
    print(f"\n{'='*70}")
