"""
Betting tracker — store and manage your actual bets.

💡 KEY CONCEPT: You make PREDICTIONS (model outputs).
You make BETS (real money decisions based on those predictions).
This module tracks the BETS so we can:
1. Show ROI (money made/lost)
2. Calibrate confidence (did 55% bets actually hit 55%?)
3. Compare model predictions vs. your betting decisions
4. Audit your picks (why did you bet on Kopitar @ -180?)

Storage: CSV (human-readable, Excel-compatible)
Columns: date, player_name, bet_amount, odds, game_date, status, actual_goals, payout
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


DATA_DIR = Path(__file__).parent / ".." / ".." / "data"
BETS_FILE = DATA_DIR / "bets.csv"


def _ensure_bets_dir():
    """Create data directory if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_bets_file():
    """Create bets CSV with headers if it doesn't exist."""
    _ensure_bets_dir()
    if not BETS_FILE.exists():
        with open(BETS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "bet_id",
                    "date_placed",
                    "player_name",
                    "team",
                    "opponent",
                    "bet_amount",
                    "odds",
                    "implied_probability",
                    "game_date",
                    "status",  # open, won, lost, push, cancelled
                    "actual_goals",
                    "payout",
                    "profit",
                    "notes",
                ],
            )
            writer.writeheader()


def calculate_implied_probability(odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -110, +150)
    
    Returns:
        Implied probability (0.0-1.0)
    
    Examples:
        -110 (even money, slightly worse) → ~52.4%
        -150 (favor, 3:2) → ~60%
        +150 (underdog, 3:2) → ~40%
    """
    if odds == 0:
        return 0.0
    
    if odds < 0:
        # Negative odds: probability = abs(odds) / (abs(odds) + 100)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Positive odds: probability = 100 / (odds + 100)
        return 100 / (odds + 100)


def calculate_payout(bet_amount: float, odds: float, won: bool) -> float:
    """
    Calculate payout for a bet.
    
    Args:
        bet_amount: Amount wagered
        odds: American odds
        won: Did the bet win?
    
    Returns:
        Total payout (includes original bet if won)
    """
    if not won:
        return 0.0
    
    if odds < 0:
        # Negative odds: payout = bet_amount * (100 / abs(odds))
        return bet_amount * (100 / abs(odds))
    else:
        # Positive odds: payout = bet_amount * (odds / 100)
        return bet_amount * (odds / 100)


class BetTracker:
    """JSON-backed betting tracker."""
    
    def __init__(self, path: Path = BETS_FILE):
        self.path = path
        self._ensure_exists()
    
    def _ensure_exists(self):
        """Create bets CSV if it doesn't exist."""
        _ensure_bets_dir()
        if not self.path.exists():
            _ensure_bets_file()
    
    def _load(self) -> list[dict]:
        """Load all bets from CSV."""
        if not self.path.exists():
            return []
        
        bets = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bets.append(row)
        return bets
    
    def _save(self, bets: list[dict]):
        """Save bets back to CSV."""
        with open(self.path, "w", newline="") as f:
            if bets:
                writer = csv.DictWriter(f, fieldnames=bets[0].keys())
                writer.writeheader()
                writer.writerows(bets)
    
    def place_bet(
        self,
        player_name: str,
        team: str,
        opponent: str,
        bet_amount: float,
        odds: float,
        game_date: str,
        notes: str = "",
    ) -> dict:
        """
        Place a new bet.
        
        Args:
            player_name: Player you're betting on (e.g., "David Pastrnak")
            team: Player's team (e.g., "BOS")
            opponent: Opponent team (e.g., "NJD")
            bet_amount: Amount wagered ($)
            odds: American odds (e.g., -110, +150)
            game_date: Game date (YYYY-MM-DD)
            notes: Optional notes (e.g., "model said 55%")
        
        Returns:
            The created bet dict
        """
        bets = self._load()
        next_id = max([int(b.get("bet_id", 0)) for b in bets], default=0) + 1
        
        implied_prob = calculate_implied_probability(odds)
        
        bet = {
            "bet_id": str(next_id),
            "date_placed": datetime.now().isoformat(),
            "player_name": player_name,
            "team": team,
            "opponent": opponent,
            "bet_amount": str(bet_amount),
            "odds": str(odds),
            "implied_probability": f"{implied_prob:.3f}",
            "game_date": game_date,
            "status": "open",
            "actual_goals": "",
            "payout": "",
            "profit": "",
            "notes": notes,
        }
        
        bets.append(bet)
        self._save(bets)
        
        return bet
    
    def grade_bet(
        self,
        bet_id: int,
        actual_goals: int,
        game_date: Optional[str] = None,
    ) -> dict:
        """
        Grade a bet against actual results.
        
        Args:
            bet_id: The bet ID to grade
            actual_goals: How many goals the player actually scored
            game_date: Optional game date override
        
        Returns:
            Updated bet dict
        """
        bets = self._load()
        
        for bet in bets:
            if int(bet["bet_id"]) == bet_id:
                # Determine win/loss
                odds = float(bet["odds"])
                bet_amount = float(bet["bet_amount"])
                
                # Anytime Scorer: did they score at least 1?
                won = actual_goals >= 1
                
                # Calculate payout
                payout = calculate_payout(bet_amount, odds, won)
                profit = payout - bet_amount
                
                # Update bet
                bet["status"] = "won" if won else "lost"
                bet["actual_goals"] = str(actual_goals)
                bet["payout"] = f"{payout:.2f}"
                bet["profit"] = f"{profit:.2f}"
                
                self._save(bets)
                return bet
        
        return {}
    
    def list_open(self, game_date: Optional[str] = None) -> list[dict]:
        """List all open bets, optionally for a specific date."""
        bets = self._load()
        open_bets = [b for b in bets if b["status"] == "open"]
        
        if game_date:
            open_bets = [b for b in open_bets if b["game_date"] == game_date]
        
        return open_bets
    
    def list_all(self, game_date: Optional[str] = None) -> list[dict]:
        """List all bets, optionally for a specific date."""
        bets = self._load()
        
        if game_date:
            bets = [b for b in bets if b["game_date"] == game_date]
        
        return bets
    
    def get_stats(self, game_date: Optional[str] = None) -> dict:
        """
        Calculate betting statistics.
        
        Returns:
            Dict with wins, losses, ROI, etc.
        """
        bets = self._load()
        
        if game_date:
            bets = [b for b in bets if b["game_date"] == game_date]
        
        # Filter to graded bets
        graded = [b for b in bets if b["status"] in ["won", "lost"]]
        
        if not graded:
            return {
                "total_bets": len(bets),
                "graded": 0,
                "wins": 0,
                "losses": 0,
                "win_pct": 0.0,
                "total_wagered": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
            }
        
        wins = sum(1 for b in graded if b["status"] == "won")
        losses = sum(1 for b in graded if b["status"] == "lost")
        
        total_wagered = sum(float(b["bet_amount"]) for b in graded)
        total_profit = sum(float(b["profit"]) for b in graded)
        
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
        
        return {
            "total_bets": len(bets),
            "graded": len(graded),
            "wins": wins,
            "losses": losses,
            "win_pct": wins / len(graded) * 100 if graded else 0.0,
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": roi,
        }
