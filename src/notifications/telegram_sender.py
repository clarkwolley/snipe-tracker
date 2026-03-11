"""
Telegram delivery for Snipe Tracker picks.

Sends a compact picks summary via Telegram Bot API.
No external libraries needed — just plain HTTPS requests.

Setup:
1. Message @BotFather on Telegram, create a bot, get the token
2. Send any message to your bot, then visit:
   https://api.telegram.org/bot<TOKEN>/getUpdates
   to find your chat_id
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""

import requests
import pandas as pd

from src.notifications.settings import load_settings


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4096  # Telegram's limit


def _format_picks_message(pred_df: pd.DataFrame, top_n: int = 15) -> str:
    """
    Format prediction data into a compact Telegram message.

    Uses Telegram's MarkdownV2 formatting for clean output.
    """
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    top = pred_df.head(top_n).copy()

    lines = [
        f"🏒 *Snipe Tracker* — {today}",
        f"_{len(pred_df)} players analyzed_",
        "",
        "🎯 *Top Picks*",
        "```",
    ]

    # Table header
    lines.append(f"{'#':>2} {'Player':<20} {'Team':>4} {'Prob':>5} {'Streak'}")
    lines.append("-" * 45)

    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row["name"][:20]
        prob = f"{row['goal_probability'] * 100:.0f}%"

        streak = ""
        if row.get("is_hot", 0):
            streak = f"🔥{int(row.get('goal_streak', 0))}"
        elif row.get("drought", 0) >= 5:
            streak = f"❄️{int(row.get('drought', 0))}"

        matchup = f"{'vs' if row['is_home'] else '@'}{row['opponent']}"
        lines.append(f"{i:>2} {name:<20} {matchup:>7} {prob:>5} {streak}")

    lines.append("```")

    # Per-game summary (compact)
    lines.append("")
    lines.append("📋 *By Game*")

    seen = set()
    for _, row in pred_df.iterrows():
        if row["is_home"]:
            key = f"{row['opponent']}@{row['team']}"
        else:
            key = f"{row['team']}@{row['opponent']}"

        if key in seen:
            continue
        seen.add(key)

        home = row["team"] if row["is_home"] else row["opponent"]
        away = row["opponent"] if row["is_home"] else row["team"]

        # Top scorer from each side
        home_top = pred_df[pred_df["team"] == home].head(1)
        away_top = pred_df[pred_df["team"] == away].head(1)

        home_pick = ""
        if not home_top.empty:
            r = home_top.iloc[0]
            home_pick = f"{r['name'].split()[-1]} {r['goal_probability']*100:.0f}%"

        away_pick = ""
        if not away_top.empty:
            r = away_top.iloc[0]
            away_pick = f"{r['name'].split()[-1]} {r['goal_probability']*100:.0f}%"

        lines.append(f"  {away} @ {home}: {home_pick} / {away_pick}")

    lines.append("")
    lines.append("_Full report sent via email_ 📧")

    return "\n".join(lines)


def _format_grade_message(graded: pd.DataFrame) -> str:
    """Format grading results into a Telegram message."""
    played = graded[graded["played"] == 1]
    date = graded["prediction_date"].iloc[0]

    total = len(played)
    actual = int(played["actual_scored"].sum())
    predicted = int(played["predicted_goal"].sum())
    hits = int(played["hit"].sum())
    precision = hits / max(predicted, 1) * 100

    lines = [
        f"📊 *Scorecard* — {date}",
        "",
        f"Players tracked: {total}",
        f"Actually scored: {actual}",
        f"Predicted goals: {predicted}",
        f"Hits: {hits}/{predicted} ({precision:.0f}% precision)",
        "",
    ]

    # Top hits
    top_hits = played[played["actual_scored"] == 1].nlargest(5, "goal_probability")
    if not top_hits.empty:
        lines.append("✅ *Top Hits*")
        for _, r in top_hits.iterrows():
            lines.append(
                f"  {r['name']} ({r['team']}) — "
                f"{r['goal_probability']*100:.0f}% → {int(r['actual_goals'])}G"
            )

    return "\n".join(lines)


def send_picks(pred_df: pd.DataFrame, top_n: int = 15) -> bool:
    """
    Send today's picks summary via Telegram.

    Args:
        pred_df: DataFrame from predict_tonight().
        top_n: Number of top picks to include.

    Returns:
        True if sent successfully, False otherwise.
    """
    settings = load_settings()
    token = settings.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = settings.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("  ⚠️  Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False

    message = _format_picks_message(pred_df, top_n)

    # Truncate if too long (shouldn't happen with top_n=15)
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH - 20] + "\n\n_(truncated)_"

    return _send_message(token, chat_id, message)


def send_grade(graded: pd.DataFrame) -> bool:
    """Send grading results via Telegram."""
    settings = load_settings()
    token = settings.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = settings.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        return False

    message = _format_grade_message(graded)
    return _send_message(token, chat_id, message)


def _send_message(token: str, chat_id: str, text: str) -> bool:
    """Send a message via Telegram Bot API."""
    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print("  ✅ Telegram message sent")
        return True
    except requests.exceptions.HTTPError as e:
        # Common issue: MarkdownV2 parsing errors. Fall back to plain text.
        if resp.status_code == 400:
            payload["parse_mode"] = None
            try:
                resp2 = requests.post(url, json=payload, timeout=15)
                resp2.raise_for_status()
                print("  ✅ Telegram message sent (plain text fallback)")
                return True
            except Exception:
                pass
        print(f"  ❌ Telegram failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")
        return False
