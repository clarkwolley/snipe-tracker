#!/usr/bin/env python3
"""
Betting CLI — Interactive interface to place and track bets.

Usage:
    python bet_cli.py place "David Pastrnak" 50 -110 2026-03-20
    python bet_cli.py list 2026-03-20
    python bet_cli.py grade 2026-03-20
    python bet_cli.py stats 2026-03-20
"""

import sys
from datetime import datetime

from src.predictions.bet_tracker import BetTracker
from src.predictions.bet_grader import grade_bets_for_date, print_bet_scorecard, compare_predictions_vs_bets
from src.notifications.telegram_bet_handler import parse_bet_command


def cmd_place(args):
    """Place a bet: place [name] [amount] [odds] [game_date] [notes]"""
    if len(args) < 4:
        print("❌ Usage: place [name] [amount] [odds] [game_date] [notes]")
        print("Example: place 'David Pastrnak' 50 -110 2026-03-20 'model said 55%'")
        return
    
    name = args[0]
    amount = float(args[1])
    odds = float(args[2])
    game_date = args[3]
    notes = " ".join(args[4:]) if len(args) > 4 else ""
    
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date: {game_date}. Use YYYY-MM-DD format.")
        return
    
    tracker = BetTracker()
    bet = tracker.place_bet(
        player_name=name,
        team="",  # Can be added later
        opponent="",
        bet_amount=amount,
        odds=odds,
        game_date=game_date,
        notes=notes,
    )
    
    print(f"✅ Bet placed!")
    print(f"   Player: {name}")
    print(f"   Amount: ${amount:.2f}")
    print(f"   Odds: {odds:+.0f}")
    print(f"   Game: {game_date}")
    print(f"   Bet ID: #{bet['bet_id']}")
    if notes:
        print(f"   Notes: {notes}")


def cmd_list(args):
    """List bets: list [game_date]"""
    game_date = args[0] if args else None
    
    tracker = BetTracker()
    bets = tracker.list_all(game_date=game_date)
    
    if not bets:
        print(f"❌ No bets found for {game_date or 'any date'}")
        return
    
    date_str = game_date or "All Time"
    print(f"\n📋 BETS — {date_str}")
    print(f"{'='*80}")
    print(f"{'ID':>3} {'Player':<25} {'Amount':>8} {'Odds':>7} {'Game':>12} {'Status':<10}")
    print(f"{'-'*80}")
    
    for bet in bets:
        print(
            f"{bet['bet_id']:>3} {bet['player_name']:<25} "
            f"${float(bet['bet_amount']):>7.2f} {float(bet['odds']):>7.0f} "
            f"{bet['game_date']:>12} {bet['status']:<10}"
        )
    
    print(f"{'='*80}")
    print(f"Total: {len(bets)} bets")


def cmd_grade(args):
    """Grade bets for a date: grade [game_date]"""
    if not args:
        print("❌ Usage: grade [game_date]")
        print("Example: grade 2026-03-20")
        return
    
    game_date = args[0]
    
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date: {game_date}. Use YYYY-MM-DD format.")
        return
    
    print_bet_scorecard(game_date)
    compare_predictions_vs_bets(game_date)


def cmd_stats(args):
    """Show betting statistics: stats [game_date]"""
    game_date = args[0] if args else None
    
    tracker = BetTracker()
    stats = tracker.get_stats(game_date=game_date)
    
    date_str = game_date or "All Time"
    
    print(f"\n📊 BETTING STATS — {date_str}")
    print(f"{'='*60}")
    print(f"  Total bets:     {stats['total_bets']}")
    print(f"  Graded:         {stats['graded']}")
    print(f"  Wins:           {stats['wins']}")
    print(f"  Losses:         {stats['losses']}")
    
    if stats['graded'] > 0:
        print(f"  Win rate:       {stats['win_pct']:.1f}%")
        print(f"  Total wagered:  ${stats['total_wagered']:.2f}")
        print(f"  Profit/Loss:    ${stats['total_profit']:+.2f}")
        print(f"  ROI:            {stats['roi']:+.1f}%")
    
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("🐕 Snipe Tracker — Betting CLI")
        print()
        print("Commands:")
        print("  place [name] [amount] [odds] [game_date] [notes] — Place a bet")
        print("  list [game_date]                              — List bets")
        print("  grade [game_date]                             — Grade bets for a date")
        print("  stats [game_date]                             — Show statistics")
        print()
        print("Examples:")
        print('  python bet_cli.py place "David Pastrnak" 50 -110 2026-03-20')
        print("  python bet_cli.py list 2026-03-20")
        print("  python bet_cli.py grade 2026-03-20")
        print("  python bet_cli.py stats 2026-03-20")
        return
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    if cmd == "place":
        cmd_place(args)
    elif cmd == "list":
        cmd_list(args)
    elif cmd == "grade":
        cmd_grade(args)
    elif cmd == "stats":
        cmd_stats(args)
    else:
        print(f"❌ Unknown command: {cmd}")
        print("Use 'python bet_cli.py' with no args to see help")


if __name__ == "__main__":
    main()
