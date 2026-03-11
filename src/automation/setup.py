"""
Interactive setup for Snipe Tracker notifications.

Walks you through configuring email and Telegram delivery,
then writes credentials to .env (which is gitignored).

Usage:
    python -m src.automation.setup
"""

import os
import sys
import time
from pathlib import Path

import requests

from src.notifications.settings import PROJECT_ROOT, ENV_FILE, load_settings


def _read_input(prompt: str, default: str = "") -> str:
    """Read input with optional default value."""
    if default:
        result = input(f"  {prompt} [{default}]: ").strip()
        return result or default
    return input(f"  {prompt}: ").strip()


def _confirm(prompt: str) -> bool:
    """Yes/no confirmation prompt."""
    return input(f"  {prompt} [Y/n]: ").strip().lower() != "n"


def _save_env(settings: dict[str, str]) -> None:
    """Write settings to .env file, preserving comments and unknown keys."""
    existing_lines: list[str] = []
    existing_keys: set[str] = set()

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                existing_lines.append(line)
                continue
            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in settings:
                    existing_lines.append(f"{key}={settings[key]}")
                    existing_keys.add(key)
                else:
                    existing_lines.append(line)  # preserve unknown keys

    # Append new keys not already in file
    for key, value in settings.items():
        if key not in existing_keys:
            existing_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(existing_lines) + "\n")


# --- Email -------------------------------------------------------------------


def setup_email() -> dict[str, str]:
    """Interactive email configuration."""
    print("\n📧 EMAIL SETUP")
    print("-" * 40)
    print("  For Gmail: enable 2FA, then create an App Password at:")
    print("  https://myaccount.google.com/apppasswords\n")

    current = load_settings()

    settings = {}
    settings["SMTP_HOST"] = _read_input("SMTP host", current.get("SMTP_HOST", "smtp.gmail.com"))
    settings["SMTP_PORT"] = _read_input("SMTP port", current.get("SMTP_PORT", "587"))
    settings["SMTP_USER"] = _read_input("Your email address", current.get("SMTP_USER", ""))
    settings["SMTP_PASSWORD"] = _read_input("App password", current.get("SMTP_PASSWORD", ""))

    default_recipient = current.get("EMAIL_RECIPIENT", settings["SMTP_USER"])
    # Don't use a bogus default if someone previously typed "Y"
    if "@" not in default_recipient:
        default_recipient = settings["SMTP_USER"]

    print("\n  Where should reports be sent?")
    print(f"  (Press Enter to send to yourself: {default_recipient})")
    recipient = _read_input("Recipient email address", default_recipient)

    # Validate it looks like an email
    while "@" not in recipient or "." not in recipient:
        print("  ⚠️  That doesn't look like an email address. Try again.")
        recipient = _read_input("Recipient email address", default_recipient)

    settings["EMAIL_RECIPIENT"] = recipient

    return settings


def test_email() -> bool:
    """Send a test email."""
    from src.notifications.email_sender import send_report

    test_html = PROJECT_ROOT / "reports" / "_test_email.html"
    test_html.parent.mkdir(exist_ok=True)
    test_html.write_text(
        "<html><body>"
        "<h1>🏒 Snipe Tracker Test</h1>"
        "<p>If you see this, email delivery is working! 🐶</p>"
        "</body></html>"
    )

    result = send_report(str(test_html), subject="🏒 Snipe Tracker — Test Email")
    test_html.unlink(missing_ok=True)
    return result


# --- Telegram ----------------------------------------------------------------


def setup_telegram() -> dict[str, str]:
    """Interactive Telegram configuration."""
    print("\n💬 TELEGRAM SETUP")
    print("-" * 40)
    print("  1. Open Telegram → search @BotFather → /newbot → get token")
    print("  2. We'll auto-detect your chat ID (you just need to message the bot)\n")

    current = load_settings()

    settings = {}
    settings["TELEGRAM_BOT_TOKEN"] = _read_input(
        "Bot token (from @BotFather)", current.get("TELEGRAM_BOT_TOKEN", "")
    )

    token = settings["TELEGRAM_BOT_TOKEN"]
    if not token:
        print("  ⚠️  No token provided, skipping Telegram setup.")
        return settings

    # Auto-detect chat ID
    print("\n  🔍 Looking for your chat ID...")
    chat_id = _detect_chat_id(token)

    if chat_id:
        print(f"  ✅ Found your chat ID: {chat_id}")
        settings["TELEGRAM_CHAT_ID"] = str(chat_id)
    else:
        print("  ⚠️  No messages found from you yet.")
        print("  👉 Open Telegram and send ANY message to your bot.")
        input("  Press Enter when you've sent a message...")

        # Try again
        chat_id = _detect_chat_id(token)
        if chat_id:
            print(f"  ✅ Found your chat ID: {chat_id}")
            settings["TELEGRAM_CHAT_ID"] = str(chat_id)
        else:
            print("  ❌ Still no messages. Enter your chat ID manually:")
            print("     Visit: https://api.telegram.org/bot<TOKEN>/getUpdates")
            settings["TELEGRAM_CHAT_ID"] = _read_input(
                "Chat ID", current.get("TELEGRAM_CHAT_ID", "")
            )

    return settings


def _detect_chat_id(token: str) -> int | None:
    """
    Auto-detect the user's chat ID from bot updates.

    Returns the chat_id of the first person who messaged the bot,
    or None if no messages found.
    """
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if not data.get("ok") or not data.get("result"):
            return None

        for update in data["result"]:
            msg = update.get("message", {})
            chat = msg.get("chat", {})
            chat_id = chat.get("id")
            if chat_id:
                return chat_id

        return None
    except Exception:
        return None


def test_telegram() -> bool:
    """Send a test Telegram message."""
    from src.notifications.telegram_sender import _send_message
    settings = load_settings()
    token = settings.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = settings.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("  ❌ Telegram not configured.")
        return False

    return _send_message(
        token, chat_id,
        "🏒 Snipe Tracker — Test message!\n\nIf you see this, Telegram is working! 🐶"
    )


# --- launchd -----------------------------------------------------------------


def setup_launchd() -> None:
    """Generate and install the macOS launchd plist."""
    print("\n⏰ LAUNCHD SETUP")
    print("-" * 40)
    print("  This schedules Snipe Tracker to run automatically every day.\n")

    venv_python = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        print(f"  ⚠️  Venv python not found at {venv_python}")
        venv_python = _read_input("Path to python", sys.executable)

    hour = _read_input(
        "Run daily at what hour? (24h format, e.g. 16 = 4 PM)", "16"
    )
    minute = _read_input("Minute", "00")

    plist_name = "com.snipetracker.daily"
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_name}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{venv_python}</string>
        <string>-m</string>
        <string>src.automation.runner</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{int(hour)}</integer>
        <key>Minute</key>
        <integer>{int(minute)}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{PROJECT_ROOT}/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{PROJECT_ROOT}/logs/launchd_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>"""

    # Create logs dir
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    # Write plist to LaunchAgents
    launch_dir = Path.home() / "Library" / "LaunchAgents"
    launch_dir.mkdir(exist_ok=True)
    plist_path = launch_dir / f"{plist_name}.plist"
    plist_path.write_text(plist_content)

    # Also save a copy in the project for reference
    local_copy = PROJECT_ROOT / f"{plist_name}.plist"
    local_copy.write_text(plist_content)

    # Auto-load it
    os.system(f"launchctl unload {plist_path} 2>/dev/null")
    os.system(f"launchctl load {plist_path}")

    print(f"\n  ✅ Plist installed and loaded!")
    print(f"     {plist_path}")
    print(f"\n  📅 Runs daily at {hour}:{minute.zfill(2)}")
    print(f"\n  Useful commands:")
    print(f"    launchctl start {plist_name}     # Run it NOW (test)")
    print(f"    launchctl unload {plist_path}    # Disable")
    print(f"    launchctl load {plist_path}      # Re-enable")
    print(f"    cat ~/Projects/snipe-tracker/logs/runner.log  # Check logs")


# --- Main --------------------------------------------------------------------


def main():
    print("\n🐶 SNIPE TRACKER — Notification Setup")
    print("=" * 50)

    # Step 1: Email
    print("\n📧 Set up email delivery?")
    if _confirm("Configure email?"):
        email_settings = setup_email()
        _save_env(email_settings)
        print("\n  Sending test email...")
        if test_email():
            print("  📧 Check your inbox!")
        else:
            print("  ⚠️  Email test failed. Check credentials and try again.")

    # Step 2: Telegram
    print("\n💬 Set up Telegram delivery?")
    if _confirm("Configure Telegram?"):
        tg_settings = setup_telegram()
        _save_env(tg_settings)
        print("\n  Sending test message...")
        if test_telegram():
            print("  💬 Check Telegram!")
        else:
            print("  ⚠️  Telegram test failed. Check token/chat ID.")

    # Step 3: Scheduler
    print("\n⏰ Set up daily automation (launchd)?")
    if _confirm("Install daily scheduler?"):
        setup_launchd()

    # Final status
    print("\n" + "=" * 50)
    print("  ✅ Setup complete! Verify with:")
    print("     python -m src.automation.runner --status")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
