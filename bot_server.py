#!/usr/bin/env python3
"""
Telegram bot webhook server for Snipe Tracker.

Runs a simple Flask app that listens for Telegram webhook updates.

Usage:
    # Development (localhost, won't work for Telegram webhooks):
    python bot_server.py --dev
    
    # Production (set webhook, listen for real updates):
    python bot_server.py --webhook https://your-domain.com/telegram
    
    # Just set webhook, don't run server:
    python bot_server.py --set-webhook https://your-domain.com/telegram

Requirements:
    pip install flask python-dotenv requests
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Check for Flask
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("❌ Flask not installed. Run: pip install flask")
    sys.exit(1)

from src.notifications.settings import load_settings
from src.notifications.telegram_webhook import handle_update, setup_webhook


def create_app():
    """Create and configure Flask app."""
    app = Flask(__name__)
    
    @app.route("/", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "message": "Snipe Tracker Bot running"}), 200
    
    @app.route("/telegram", methods=["POST"])
    def telegram():
        """Telegram webhook endpoint."""
        try:
            update = request.get_json()
            
            # Log for debugging
            print(f"\n📨 Received Telegram update: {json.dumps(update, indent=2)}")
            
            # Process the update
            response = handle_update(update)
            
            print(f"✅ Processed successfully")
            
            return jsonify(response), 200
        
        except Exception as e:
            print(f"❌ Error processing update: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(e)}), 500
    
    @app.route("/status", methods=["GET"])
    def status():
        """Check bot status."""
        settings = load_settings()
        
        has_token = bool(settings.get("telegram_bot_token"))
        has_chat = bool(settings.get("telegram_chat_id"))
        
        return jsonify({
            "running": True,
            "telegram_configured": has_token and has_chat,
            "has_token": has_token,
            "has_chat_id": has_chat,
        }), 200
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Snipe Tracker Telegram Bot Server")
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (localhost, no webhooks)"
    )
    parser.add_argument(
        "--webhook",
        type=str,
        help="Set webhook URL and run server (e.g., https://my-domain.com/telegram)"
    )
    parser.add_argument(
        "--set-webhook",
        type=str,
        help="Only set webhook URL, don't run server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run server on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings()
    bot_token = settings.get("telegram_bot_token")
    
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not configured in .env")
        print("   Add to .env:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token_here")
        sys.exit(1)
    
    # Handle webhook setup
    if args.set_webhook:
        print(f"\n🔧 Setting Telegram webhook to: {args.set_webhook}")
        success = setup_webhook(bot_token, args.set_webhook)
        sys.exit(0 if success else 1)
    
    # Create app
    app = create_app()
    
    # Show banner
    print("\n" + "="*70)
    print("🐕 SNIPE TRACKER TELEGRAM BOT")
    print("="*70)
    print(f"\n✅ Bot token configured")
    print(f"✅ Settings loaded from .env")
    
    if args.webhook:
        print(f"\n🔧 Setting webhook to: {args.webhook}")
        success = setup_webhook(bot_token, args.webhook)
        if not success:
            print("❌ Failed to set webhook. Check your URL and internet connection.")
            sys.exit(1)
    
    # Print usage
    if args.dev:
        print("\n📍 Running in DEVELOPMENT mode (localhost only)")
        print("   Telegram can't reach localhost webhooks!")
        print("   Use --webhook to set a public URL for production\n")
    else:
        print(f"\n📍 Running on {args.host}:{args.port}")
        if args.webhook:
            print(f"   Webhook: {args.webhook}")
        print("   Ready to receive bot commands!\n")
    
    # Print available commands
    print("📋 Available commands (send to bot):")
    print("   /bet Name Amount Odds Date [Notes]  — Place a bet")
    print("   /bets [Date]                         — Show open bets")
    print("   /stats                               — Show statistics")
    print("   /grade Date                          — Grade bets for date")
    print("   /help                                — Show help\n")
    
    print("🌐 Health check: http://localhost:5000/")
    print("📊 Status check: http://localhost:5000/status")
    print("="*70 + "\n")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.dev,
        use_reloader=args.dev,
    )


if __name__ == "__main__":
    main()
