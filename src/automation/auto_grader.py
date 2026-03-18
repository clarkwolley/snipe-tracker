"""
Automatic grading system for Snipe Tracker.

Runs after games finish (evening). Automatically:
1. Fetches actual game results
2. Grades all open bets
3. Calculates ROI, hit rate
4. Compares your picks vs model's predictions
5. Sends summary via Telegram

No manual commands needed — fully automated!

Usage:
    from src.automation.auto_grader import auto_grade_bets
    
    results = auto_grade_bets(game_date="2026-03-20")
    print(f"Graded {results['graded']} bets")
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from src.predictions.bet_tracker import BetTracker
from src.predictions.bet_grader import grade_bets_for_date


logger = logging.getLogger(__name__)


def auto_grade_bets(game_date: str = None) -> Dict[str, Any]:
    """
    Automatically grade all bets for a given date.
    
    Args:
        game_date: Game date (YYYY-MM-DD). If None, uses yesterday.
    
    Returns:
        Dict with grading results:
        {
            'game_date': '2026-03-20',
            'graded': 5,
            'wins': 3,
            'losses': 2,
            'wagered': 195.00,
            'profit': -52.55,
            'roi': -26.9,
            'bets': [
                {
                    'player_name': 'Pastrnak',
                    'bet_amount': 50,
                    'result': 'won',
                    'actual_goals': 2,
                    'profit': 45.45
                },
                ...
            ],
            'model_comparison': {
                'top_model_picks': [...],
                'your_picks': [...],
                'divergences': [...]
            }
        }
    """
    if game_date is None:
        # Default to yesterday (games finished)
        game_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"\n📊 AUTO-GRADING BETS")
    logger.info(f"   Game date: {game_date}")
    
    # Grade bets for this date
    results = grade_bets_for_date(game_date)
    
    logger.info(f"   Graded: {results.get('graded', 0)}")
    logger.info(f"   Record: {results.get('wins', 0)}-{results.get('losses', 0)}")
    logger.info(f"   ROI: {results.get('roi', 0):+.1f}%")
    
    return results


def format_grading_summary(results: Dict[str, Any]) -> str:
    """
    Format grading results for Telegram notification.
    
    Args:
        results: Dict from auto_grade_bets()
    
    Returns:
        Formatted message string
    """
    graded = results.get("graded", 0)
    if graded == 0:
        return f"No bets to grade for {results.get('game_date', 'today')}"
    
    wins = results.get("wins", 0)
    losses = results.get("losses", 0)
    wagered = results.get("wagered", 0)
    profit = results.get("profit", 0)
    roi = results.get("roi", 0)
    
    lines = [
        f"🎲 BET GRADING — {results.get('game_date', 'today')}",
        f"",
        f"Record: {wins}-{losses} ({wins*100//(wins+losses) if (wins+losses) > 0 else 0}%)",
        f"Wagered: ${wagered:.2f}",
        f"Profit: ${profit:+.2f}",
        f"ROI: {roi:+.1f}%",
    ]
    
    # Show individual bets
    bets = results.get("bets", [])
    if bets:
        lines.append("")
        lines.append("Details:")
        for bet in bets:
            result_emoji = "✅" if bet.get("result") == "won" else "❌"
            lines.append(
                f"  {result_emoji} {bet.get('player_name', 'Unknown')} "
                f"(${bet.get('bet_amount', 0)}) → "
                f"${bet.get('profit', 0):+.2f}"
            )
    
    # Model comparison
    model_comp = results.get("model_comparison", {})
    if model_comp:
        your_hits = model_comp.get("your_hit_rate", 0)
        model_hits = model_comp.get("model_hit_rate", 0)
        
        lines.append("")
        lines.append("vs Model:")
        lines.append(f"  Your hit rate: {your_hits:.0%}")
        lines.append(f"  Model hit rate: {model_hits:.0%}")
        
        if your_hits > model_hits:
            lines.append(f"  🔥 You beat the model!")
        elif your_hits < model_hits:
            lines.append(f"  📚 Model was better today")
    
    return "\n".join(lines)



