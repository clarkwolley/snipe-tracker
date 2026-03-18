"""
Telegram command handler for placing and tracking bets.

💡 USAGE:
/bet Pastrnak 20 -110 2026-03-20 "model said 55%"

Formats:
/bet [name] [amount] [odds] [game_date] [notes]

Examples:
/bet Pastrnak 20 -110 2026-03-20
/bet "David Pastrnak" 50 +150 2026-03-20 "high confidence play"
/bet Malkin 25 -120 2026-03-20

This handler:
1. Parses incoming /bet commands
2. Validates inputs
3. Stores the bet
4. Sends confirmation back to Telegram
5. Logs for audit trail
"""

import re
from datetime import datetime
from typing import Optional, Tuple
import requests

from src.notifications.settings import load_settings
from src.predictions.bet_tracker import BetTracker, calculate_implied_probability


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def parse_bet_command(text: str) -> Optional[dict]:
    """
    Parse a /bet command.
    
    Format: /bet [name] [amount] [odds] [game_date] [notes]
    
    Args:
        text: Raw message text (e.g., "/bet Pastrnak 20 -110 2026-03-20")
    
    Returns:
        Dict with parsed fields, or None if parse failed
    
    Examples:
        "/bet Pastrnak 20 -110 2026-03-20"
        → {"name": "Pastrnak", "amount": 20.0, "odds": -110.0, "game_date": "2026-03-20", "notes": ""}
        
        "/bet \"David Pastrnak\" 50 +150 2026-03-20 model said 55%"
        → {"name": "David Pastrnak", "amount": 50.0, "odds": 150.0, "game_date": "2026-03-20", "notes": "model said 55%"}
    """
    if not text.startswith("/bet"):
        return None
    
    # Remove /bet command
    content = text[4:].strip()
    
    # Try to parse: name amount odds game_date [notes]
    # Handle quoted names like "David Pastrnak"
    
    parts = []
    in_quotes = False
    current = ""
    
    for char in content:
        if char == '"':
            in_quotes = not in_quotes
        elif char == " " and not in_quotes:
            if current:
                parts.append(current)
                current = ""
        else:
            current += char
    
    if current:
        parts.append(current)
    
    if len(parts) < 4:
        return None
    
    try:
        name = parts[0]
        amount = float(parts[1])
        odds = float(parts[2])
        game_date = parts[3]
        notes = " ".join(parts[4:]) if len(parts) > 4 else ""
        
        # Validate game_date format (YYYY-MM-DD)
        datetime.strptime(game_date, "%Y-%m-%d")
        
        # Validate amount and odds
        if amount <= 0:
            return None
        
        return {
            "name": name,
            "amount": amount,
            "odds": odds,
            "game_date": game_date,
            "notes": notes,
        }
    except (ValueError, IndexError):
        return None


def send_telegram_message(chat_id: str, message: str, token: str) -> bool:
    """
    Send a message via Telegram Bot API.
    
    Args:
        chat_id: Telegram chat ID
        message: Message text
        token: Telegram bot token
    
    Returns:
        True if sent successfully
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, data=data, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False


def handle_bet_command(
    command_text: str,
    player_team: str = "",
    opponent_team: str = "",
    chat_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Handle a /bet command.
    
    Args:
        command_text: Raw /bet command text
        player_team: Player's team (e.g., "BOS"). If blank, ignored.
        opponent_team: Opponent team (e.g., "NJD"). If blank, ignored.
        chat_id: Telegram chat ID for sending confirmation
    
    Returns:
        (success: bool, message: str)
    """
    # Parse the command
    parsed = parse_bet_command(command_text)
    
    if not parsed:
        return False, "❌ Invalid format. Use: /bet [name] [amount] [odds] [game_date] [notes]\nExample: /bet Pastrnak 20 -110 2026-03-20"
    
    # Place the bet
    tracker = BetTracker()
    
    bet = tracker.place_bet(
        player_name=parsed["name"],
        team=player_team,
        opponent=opponent_team,
        bet_amount=parsed["amount"],
        odds=parsed["odds"],
        game_date=parsed["game_date"],
        notes=parsed["notes"],
    )
    
    # Format confirmation message
    implied_prob = float(bet["implied_probability"])
    message = (
        f"✅ *Bet Placed*\n\n"
        f"Player: {parsed['name']}\n"
        f"Amount: ${parsed['amount']:.2f}\n"
        f"Odds: {parsed['odds']:+.0f}\n"
        f"Implied: {implied_prob*100:.1f}%\n"
        f"Game: {parsed['game_date']}\n"
        f"Bet ID: #{bet['bet_id']}\n"
    )
    
    if parsed["notes"]:
        message += f"\nNotes: {parsed['notes']}"
    
    # Send Telegram confirmation if chat_id provided
    if chat_id:
        try:
            settings = load_settings()
            token = settings.get("telegram_bot_token")
            if token:
                send_telegram_message(chat_id, message, token)
        except Exception as e:
            print(f"⚠️  Couldn't send Telegram confirmation: {e}")
    
    return True, message


def format_bets_summary(game_date: Optional[str] = None) -> str:
    """
    Format a summary of bets for a given game date.
    
    Args:
        game_date: Optional date filter (YYYY-MM-DD)
    
    Returns:
        Formatted message for Telegram
    """
    tracker = BetTracker()
    
    open_bets = tracker.list_open(game_date=game_date)
    all_bets = tracker.list_all(game_date=game_date)
    stats = tracker.get_stats(game_date=game_date)
    
    if not all_bets:
        return "📊 No bets recorded yet."
    
    date_str = game_date or "All Time"
    
    message = f"📊 *Betting Summary* — {date_str}\n\n"
    
    # Stats
    message += f"Bets: {stats['graded']}/{stats['total_bets']} graded\n"
    message += f"Record: {stats['wins']}-{stats['losses']} ({stats['win_pct']:.1f}% win rate)\n"
    message += f"Wagered: ${stats['total_wagered']:.2f}\n"
    message += f"Profit: ${stats['total_profit']:+.2f}\n"
    message += f"ROI: {stats['roi']:+.1f}%\n\n"
    
    # Open bets
    if open_bets:
        message += "🔴 *Open Bets*\n"
        for bet in open_bets[:5]:  # Show top 5
            message += (
                f"  {bet['player_name']}: "
                f"${bet['bet_amount']} @ {bet['odds']:+.0f} "
                f"(#{bet['bet_id']})\n"
            )
        if len(open_bets) > 5:
            message += f"  ... and {len(open_bets) - 5} more\n"
    
    return message
