"""
Settings loader for notification credentials.

Reads from .env file in the project root. Never hardcode secrets.
Falls back to environment variables if .env doesn't exist.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_settings() -> dict[str, str]:
    """
    Load notification settings from .env file or environment.

    .env format (one per line):
        SMTP_USER=you@gmail.com
        SMTP_PASSWORD=your-app-password
        EMAIL_RECIPIENT=you@gmail.com
        TELEGRAM_BOT_TOKEN=123456:ABCdef...
        TELEGRAM_CHAT_ID=123456789

    Returns:
        Dict of setting name → value.
    """
    settings: dict[str, str] = {}

    # Load from .env file first
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                settings[key.strip()] = value.strip().strip("'\"")

    # Environment variables override .env (useful for CI/testing)
    env_keys = [
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
        "EMAIL_RECIPIENT", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    ]
    for key in env_keys:
        env_val = os.environ.get(key)
        if env_val:
            settings[key] = env_val

    return settings


def is_email_configured() -> bool:
    """Check if email settings are present."""
    s = load_settings()
    return all(s.get(k) for k in ["SMTP_USER", "SMTP_PASSWORD", "EMAIL_RECIPIENT"])


def is_telegram_configured() -> bool:
    """Check if Telegram settings are present."""
    s = load_settings()
    return all(s.get(k) for k in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
