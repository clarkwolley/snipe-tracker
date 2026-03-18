"""
Automatic betting system for Snipe Tracker.

Runs after predictions are generated. Automatically places bets on top picks
based on configurable criteria (min probability, max bets per day, etc.)

No manual commands needed — fully automated!

Usage:
    from src.automation.auto_bettor import auto_place_bets
    
    predictions_df = model.predict(...)
    bets = auto_place_bets(predictions_df)
    print(f"Placed {len(bets)} bets")
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any

from src.predictions.bet_tracker import BetTracker


logger = logging.getLogger(__name__)


class BettingConfig:
    """Configuration for automatic betting."""
    
    # THRESHOLD: Only bet on players above this probability
    MIN_PROBABILITY = 0.50  # 50%+
    
    # LIMITS: Max bets per day (avoid overexposure)
    MAX_BETS_PER_DAY = 10
    
    # BET SIZING: Base amount to bet
    # Scales with confidence: low prob = smaller bet, high prob = bigger bet
    BASE_BET_AMOUNT = 50  # dollars
    
    # CONFIDENCE SCALING
    # At 50% → $30 (less confident)
    # At 55% → $50 (normal)
    # At 60% → $75 (more confident)
    # At 65%+ → $100 (very confident)
    CONFIDENCE_TIERS = {
        0.50: 30,   # 50-52%
        0.53: 50,   # 53-55%
        0.56: 75,   # 56-58%
        0.59: 100,  # 59%+
    }
    
    # PDO SAFETY: Reduce bet size if player has high PDO (lucky)
    MAX_PDO_FOR_FULL_BET = 1.05  # 105% → lucky, reduce bet
    PDO_REDUCTION = 0.6  # Multiply bet by 0.6 if PDO too high
    
    # CORRELATED BETS: Max players from same team per day
    MAX_SAME_TEAM = 3
    
    # ODDS ASSUMPTION: Default American odds (if not provided)
    DEFAULT_ODDS = -110  # Standard for player props


def _get_bet_amount(probability: float, pdo: float = 1.0, xg_per_game: float = 0.0) -> int:
    """
    Calculate bet amount based on probability and player stats.
    
    Args:
        probability: Goal probability (0.0-1.0)
        pdo: Player's PDO (1.0 = normal, >1.05 = lucky)
        xg_per_game: Expected goals per game (for context)
    
    Returns:
        Bet amount in dollars
    """
    # Start with confidence tier
    bet = BettingConfig.BASE_BET_AMOUNT
    
    for prob_threshold in sorted(BettingConfig.CONFIDENCE_TIERS.keys(), reverse=True):
        if probability >= prob_threshold:
            bet = BettingConfig.CONFIDENCE_TIERS[prob_threshold]
            break
    
    # Reduce if player is "lucky" (high PDO)
    if pdo > BettingConfig.MAX_PDO_FOR_FULL_BET:
        logger.info(f"  PDO reduction: {pdo:.2f} > {BettingConfig.MAX_PDO_FOR_FULL_BET}, reducing bet")
        bet = int(bet * BettingConfig.PDO_REDUCTION)
    
    return max(bet, 20)  # Minimum $20 bet


def auto_place_bets(
    predictions_df: pd.DataFrame,
    game_date: str = None,
    config: BettingConfig = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Automatically place bets on top predictions.
    
    Args:
        predictions_df: DataFrame with columns:
            - name: Player name
            - team: Player's team
            - opponent: Opponent team
            - goal_probability: Predicted goal probability
            - pdo (optional): Player's PDO (luck measure)
            - xg_per_game (optional): Expected goals per game
        
        game_date: Game date (YYYY-MM-DD). If None, uses today.
        
        config: BettingConfig instance. If None, uses defaults.
        
        dry_run: If True, simulate without storing bets.
    
    Returns:
        List of placed bets (dicts with bet details)
    """
    if config is None:
        config = BettingConfig()
    
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")
    
    # Filter by probability threshold
    eligible = predictions_df[predictions_df["goal_probability"] >= config.MIN_PROBABILITY].copy()
    
    if eligible.empty:
        logger.info(f"  No picks above {config.MIN_PROBABILITY*100:.0f}% threshold")
        return []
    
    # Sort by probability (descending)
    eligible = eligible.sort_values("goal_probability", ascending=False)
    
    # Limit by max bets per day
    eligible = eligible.head(config.MAX_BETS_PER_DAY)
    
    # Check team limits (avoid stacking same team too much)
    team_counts = {}
    bets_to_place = []
    
    for _, row in eligible.iterrows():
        team = row["team"]
        
        # Skip if we've already bet too many from this team
        if team_counts.get(team, 0) >= config.MAX_SAME_TEAM:
            logger.info(f"  Skipping {row['name']} ({team}) — already {config.MAX_SAME_TEAM} bets from team")
            continue
        
        team_counts[team] = team_counts.get(team, 0) + 1
        
        # Calculate bet amount
        pdo = row.get("pdo", 1.0)
        xg = row.get("xg_per_game", 0.0)
        bet_amount = _get_bet_amount(row["goal_probability"], pdo, xg)
        
        # Prepare bet data
        bet = {
            "player_name": row["name"],
            "team": row["team"],
            "opponent": row["opponent"],
            "bet_amount": bet_amount,
            "odds": config.DEFAULT_ODDS,
            "game_date": game_date,
            "probability": row["goal_probability"],
            "pdo": pdo,
            "notes": f"Auto-placed. P={row['goal_probability']*100:.1f}% PDO={pdo:.2f}",
        }
        
        bets_to_place.append(bet)
    
    if not bets_to_place:
        logger.info(f"  No bets placed (filtered by team limits)")
        return []
    
    # Place bets
    placed_bets = []
    tracker = BetTracker()
    
    for bet_data in bets_to_place:
        if dry_run:
            logger.info(f"  [DRY RUN] Would place: {bet_data['player_name']} ${bet_data['bet_amount']}")
            placed_bets.append(bet_data)
        else:
            try:
                placed_bet = tracker.place_bet(
                    player_name=bet_data["player_name"],
                    team=bet_data["team"],
                    opponent=bet_data["opponent"],
                    bet_amount=bet_data["bet_amount"],
                    odds=bet_data["odds"],
                    game_date=bet_data["game_date"],
                    notes=bet_data["notes"],
                )
                logger.info(
                    f"  ✅ Bet placed: {bet_data['player_name']} ${bet_data['bet_amount']} "
                    f"@ {bet_data['odds']} ({bet_data['probability']*100:.1f}%)"
                )
                placed_bets.append(placed_bet)
            except Exception as e:
                logger.error(f"  ❌ Failed to place bet for {bet_data['player_name']}: {e}")
    
    logger.info(f"\n📊 AUTO-BETTING SUMMARY")
    logger.info(f"   Game date: {game_date}")
    logger.info(f"   Eligible picks: {len(eligible)}")
    logger.info(f"   Placed bets: {len(placed_bets)}")
    logger.info(f"   Total wagered: ${sum(b.get('bet_amount', 0) for b in placed_bets)}")
    
    return placed_bets


def format_betting_summary(placed_bets: List[Dict[str, Any]]) -> str:
    """
    Format betting results for Telegram notification.
    
    Args:
        placed_bets: List of placed bets
    
    Returns:
        Formatted message string
    """
    if not placed_bets:
        return "No bets placed today (below probability threshold)"
    
    total_wagered = sum(b.get("bet_amount", 0) for b in placed_bets)
    avg_prob = sum(b.get("probability", 0) for b in placed_bets) / len(placed_bets)
    
    lines = [
        f"🎲 AUTO-BETTING SUMMARY",
        f"",
        f"Bets placed: {len(placed_bets)}",
        f"Total wagered: ${total_wagered}",
        f"Avg probability: {avg_prob*100:.1f}%",
        f"",
        f"Picks:",
    ]
    
    for bet in placed_bets:
        lines.append(
            f"  • {bet['player_name']} ({bet['team']}) "
            f"${bet['bet_amount']} @ {bet['odds']} "
            f"({bet['probability']*100:.0f}%)"
        )
    
    return "\n".join(lines)
