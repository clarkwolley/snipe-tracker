"""
Email delivery for Snipe Tracker reports.

Sends the full HTML prediction report via SMTP (Gmail, etc.).
Credentials come from environment variables — never hardcoded.

Setup for Gmail:
1. Enable 2FA on your Google account
2. Create an App Password: https://myaccount.google.com/apppasswords
3. Set SMTP_PASSWORD to that app password (NOT your real password)
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from src.notifications.settings import load_settings


def _build_message(
    sender: str,
    recipient: str,
    subject: str,
    html_path: str,
) -> MIMEMultipart:
    """Build a MIME email with the HTML report as the body."""
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Plain-text fallback for email clients that don't render HTML
    plain = (
        f"{subject}\n\n"
        "Your daily Snipe Tracker report is attached as HTML.\n"
        "Open in a browser for the best experience. 🏒"
    )
    msg.attach(MIMEText(plain, "plain"))

    # HTML body — inline the full report
    html_content = Path(html_path).read_text()
    msg.attach(MIMEText(html_content, "html"))

    return msg


def send_report(html_path: str, subject: str | None = None) -> bool:
    """
    Send the HTML report via email.

    Args:
        html_path: Path to the generated HTML report file.
        subject: Optional custom subject line.

    Returns:
        True if sent successfully, False otherwise.
    """
    settings = load_settings()

    smtp_host = settings.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(settings.get("SMTP_PORT", "587"))
    smtp_user = settings.get("SMTP_USER", "")
    smtp_password = settings.get("SMTP_PASSWORD", "")
    recipient = settings.get("EMAIL_RECIPIENT", "")

    if not all([smtp_user, smtp_password, recipient]):
        print("  ⚠️  Email not configured. Set SMTP_USER, SMTP_PASSWORD, EMAIL_RECIPIENT.")
        return False

    if not os.path.exists(html_path):
        print(f"  ⚠️  Report file not found: {html_path}")
        return False

    if subject is None:
        from datetime import datetime
        subject = f"🏒 Snipe Tracker Picks — {datetime.now().strftime('%Y-%m-%d')}"

    msg = _build_message(smtp_user, recipient, subject, html_path)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print(f"  ✅ Email sent to {recipient}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("  ❌ Email auth failed. Check SMTP_USER and SMTP_PASSWORD.")
        print("     For Gmail: use an App Password, not your real password.")
        return False
    except Exception as e:
        print(f"  ❌ Email failed: {e}")
        return False
