"""
Telegram webhook handler for receiving bot commands.

Listens for incoming /bet commands and processes them.

Setup:
1. Create a Flask app and register this as a webhook endpoint
2. Tell Telegram to send updates to your webhook URL
3. When user sends /bet command, this handler processes it

Usage:
    from src.notifications.telegram_webhook import setup_webhook, handle_update
    
    # In your Flask app:
    @app.route("/telegram", methods=["POST"])
    def telegram_webhook():
        update = request.get_json()
        return handle_update(update)
"""

import json
from typing import Optional, Dict, Any

from src.notifications.settings import load_settings
from src.notifications.telegram_bet_handler import (
    handle_bet_command,
    format_bets_summary,
    send_telegram_message,
)


def parse_telegram_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse incoming Telegram update.
    
    Returns dict with:
    - command: '/bet', '/bets', '/stats', etc.
    - args: List of arguments
    - chat_id: User's chat ID
    - user_id: User's ID
    - message_text: Full message text
    
    Or None if not a command.
    """
    # Telegram sends message updates in "message" key
    msg = update.get("message", {})
    if not msg:
        return None
    
    text = msg.get("text", "").strip()
    if not text:
        return None
    
    # Parse command
    parts = text.split(None, 1)
    command = parts[0] if parts else ""
    args_str = parts[1] if len(parts) > 1 else ""
    
    if not command.startswith("/"):
        return None  # Not a command
    
    return {
        "command": command.lower(),  # /bet → /bet
        "args_str": args_str,
        "chat_id": msg.get("chat", {}).get("id"),
        "user_id": msg.get("from", {}).get("id"),
        "message_text": text,
    }


def handle_update(update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle incoming Telegram webhook update.
    
    Returns JSON response for Telegram (just acknowledgement, or with inline keyboard for next steps).
    """
    parsed = parse_telegram_update(update)
    
    if not parsed:
        # Not a command, ignore
        return {"ok": True}
    
    command = parsed["command"]
    chat_id = parsed["chat_id"]
    args_str = parsed["args_str"]
    message_text = parsed["message_text"]
    
    settings = load_settings()
    bot_token = settings.get("telegram_bot_token")
    
    # Route based on command
    if command == "/bet":
        success, message = handle_bet_command(
            command_text=message_text,
            player_team="",
            opponent_team="",
            chat_id=chat_id,
        )
        
        if bot_token and chat_id:
            send_telegram_message(chat_id, message, bot_token)
        
        return {"ok": True}
    
    elif command == "/bets":
        # Show open bets
        game_date = args_str.strip() if args_str else None
        message = format_bets_summary(game_date=game_date)
        
        if bot_token and chat_id:
            send_telegram_message(chat_id, message, bot_token)
        
        return {"ok": True}
    
    elif command == "/stats":
        # Show betting stats
        from src.predictions.bet_tracker import BetTracker
        
        tracker = BetTracker()
        stats = tracker.get_stats()
        
        message = (
            f"📊 *Your Betting Stats*\n\n"
            f"Total Bets: {stats['total_bets']}\n"
            f"Graded: {stats['graded']}\n"
            f"Record: {stats['wins']}-{stats['losses']}\n"
        )
        
        if stats['graded'] > 0:
            message += (
                f"Win Rate: {stats['win_pct']:.1f}%\n"
                f"Wagered: ${stats['total_wagered']:.2f}\n"
                f"Profit: ${stats['total_profit']:+.2f}\n"
                f"ROI: {stats['roi']:+.1f}%\n"
            )
        
        if bot_token and chat_id:
            send_telegram_message(chat_id, message, bot_token)
        
        return {"ok": True}
    
    elif command == "/grade":
        # Grade bets for a specific date
        game_date = args_str.strip() if args_str else None
        
        if not game_date:
            msg = "❌ Usage: `/grade 2026-03-20`"
        else:
            from src.predictions.bet_grader import grade_bets_for_date
            
            results = grade_bets_for_date(game_date)
            
            msg = (
                f"💰 *Bets for {game_date}*\n\n"
                f"Record: {results['wins']}-{results['losses']}\n"
                f"Wagered: ${results['total_wagered']:.2f}\n"
                f"Profit: ${results['profit']:+.2f}\n"
                f"ROI: {results['roi']:+.1f}%\n"
            )
        
        if bot_token and chat_id:
            send_telegram_message(chat_id, msg, bot_token)
        
        return {"ok": True}
    
    elif command == "/help":
        help_text = (
            "*Snipe Tracker Betting Commands*\n\n"
            "`/bet Name Amount Odds Date [Notes]`\n"
            "  Place a bet\n"
            "  Example: `/bet Pastrnak 50 -110 2026-03-20`\n\n"
            "`/bets [Date]`\n"
            "  Show open bets (optional: for specific date)\n\n"
            "`/stats`\n"
            "  Show your betting statistics (wins, losses, ROI)\n\n"
            "`/grade Date`\n"
            "  Grade bets for a specific date\n"
            "  Example: `/grade 2026-03-20`\n\n"
            "`/help`\n"
            "  Show this help message\n"
        )
        
        if bot_token and chat_id:
            send_telegram_message(chat_id, help_text, bot_token)
        
        return {"ok": True}
    
    else:
        # Unknown command
        msg = f"❌ Unknown command: {command}\nUse `/help` for available commands."
        
        if bot_token and chat_id:
            send_telegram_message(chat_id, msg, bot_token)
        
        return {"ok": True}


def setup_webhook(bot_token: str, webhook_url: str) -> bool:
    """
    Register webhook URL with Telegram.
    
    Args:
        bot_token: Your Telegram bot token
        webhook_url: Full URL where Telegram should send updates
                     (e.g., https://your-domain.com/telegram)
    
    Returns:
        True if successful
    
    Example:
        setup_webhook(
            "123456:ABCdef...",
            "https://myserver.com/telegram"
        )
    """
    import requests
    
    url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    data = {"url": webhook_url}
    
    try:
        resp = requests.post(url, json=data, timeout=10)
        result = resp.json()
        
        if result.get("ok"):
            print(f"✅ Webhook set to: {webhook_url}")
            return True
        else:
            print(f"❌ Webhook setup failed: {result.get('description')}")
            return False
    except Exception as e:
        print(f"❌ Error setting webhook: {e}")
        return False
