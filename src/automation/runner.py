"""
Daily automation runner for Snipe Tracker.

Orchestrates the full daily pipeline:
1. Grade yesterday's predictions (if any)
2. Generate today's predictions
3. Deliver results via email + Telegram

Usage:
    python -m src.automation.runner              # Full daily run
    python -m src.automation.runner --predict     # Predictions only
    python -m src.automation.runner --grade       # Grade yesterday only
    python -m src.automation.runner --notify      # Re-send latest report
    python -m src.automation.runner --status      # Check config status
"""

import os
import sys
import glob
import logging
from datetime import datetime, timedelta

# --- Logging (launchd captures stdout/stderr, but let's be tidy) ---

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")


def _setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "runner.log")

    logger = logging.getLogger("snipe-tracker")
    logger.setLevel(logging.INFO)

    # File handler (append mode — keeps history)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Also log to stdout (launchd captures this)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


# --- Pipeline steps ----------------------------------------------------------


def step_grade_yesterday(log: logging.Logger) -> None:
    """Grade yesterday's predictions against actual results."""
    from src.predictions.tracker import grade_predictions, save_graded, print_scorecard
    from src.notifications.telegram_sender import send_grade

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"📊 Grading yesterday's predictions ({yesterday})...")

    graded = grade_predictions(yesterday)
    if graded.empty:
        log.info("   No predictions to grade for yesterday.")
        return

    print_scorecard(graded)
    save_graded(graded)

    # Send grade via Telegram
    try:
        send_grade(graded)
    except Exception as e:
        log.warning(f"   Telegram grade notification failed: {e}")


def step_predict_today(log: logging.Logger) -> tuple:
    """
    Generate today's predictions and save the HTML report.

    Returns:
        (pred_df, game_df, report_path) — or (None, None, None) if no games.
    """
    from src.predictions.daily import (
        predict_tonight,
        print_top_picks,
        predict_game_winners,
        print_game_picks,
    )
    from src.predictions.tracker import save_predictions
    from src.predictions.report import generate_html_report

    log.info("🏒 Generating today's predictions...")

    pred_df = predict_tonight()
    if pred_df.empty:
        log.info("   No games scheduled today. Nothing to predict.")
        return None, None, None

    print_top_picks(pred_df)

    # Game winner predictions
    game_df = predict_game_winners()
    if not game_df.empty:
        print_game_picks(game_df)

    # Save to ledger
    save_predictions(pred_df)

    # Generate HTML report
    report_path = generate_html_report(pred_df, game_df=game_df)
    log.info(f"   Report saved: {report_path}")

    return pred_df, game_df, report_path


def step_notify(
    log: logging.Logger,
    pred_df=None,
    report_path: str | None = None,
) -> None:
    """Send predictions via email and Telegram."""
    from src.notifications.email_sender import send_report
    from src.notifications.telegram_sender import send_picks
    from src.notifications.settings import is_email_configured, is_telegram_configured

    if pred_df is None or pred_df.empty:
        log.info("   No predictions to deliver.")
        return

    # Find the report if not provided
    if report_path is None:
        report_path = _find_latest_report()

    log.info("📬 Delivering results...")

    # Email: send full HTML report
    if is_email_configured():
        try:
            send_report(report_path)
        except Exception as e:
            log.error(f"   Email delivery failed: {e}")
    else:
        log.info("   ⏭️  Email not configured, skipping.")

    # Telegram: send compact summary
    if is_telegram_configured():
        try:
            send_picks(pred_df)
        except Exception as e:
            log.error(f"   Telegram delivery failed: {e}")
    else:
        log.info("   ⏭️  Telegram not configured, skipping.")


def _find_latest_report() -> str | None:
    """Find the most recent HTML report file."""
    pattern = os.path.join(REPORT_DIR, "picks_*.html")
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


# --- Status check ------------------------------------------------------------


def check_status() -> None:
    """Print configuration status for all systems."""
    from src.notifications.settings import (
        is_email_configured,
        is_telegram_configured,
        load_settings,
    )

    print("\n🐶 Snipe Tracker — System Status")
    print("=" * 50)

    # Email
    email_ok = is_email_configured()
    print(f"  📧 Email:    {'✅ configured' if email_ok else '❌ not configured'}")

    # Telegram
    tg_ok = is_telegram_configured()
    print(f"  💬 Telegram: {'✅ configured' if tg_ok else '❌ not configured'}")

    # Model
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "goal_model.pkl"
    )
    model_ok = os.path.exists(model_path)
    print(f"  🤖 Model:    {'✅ trained' if model_ok else '❌ not trained'}")

    # Data
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "game_log.csv"
    )
    data_ok = os.path.exists(data_path)
    print(f"  📊 Data:     {'✅ collected' if data_ok else '❌ no data'}")

    # Latest report
    latest = _find_latest_report()
    if latest:
        print(f"  📄 Latest:   {os.path.basename(latest)}")
    else:
        print("  📄 Latest:   no reports yet")

    # Logs
    log_file = os.path.join(LOG_DIR, "runner.log")
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"  📝 Log:      {size / 1024:.1f} KB")

    print("=" * 50)

    if not email_ok or not tg_ok:
        print("\n  💡 Run: python -m src.automation.setup")
        print("     to configure notifications.\n")


# --- Main entry point --------------------------------------------------------


def run(mode: str = "full") -> None:
    """
    Execute the daily pipeline.

    Args:
        mode: 'full', 'predict', 'grade', 'notify', or 'status'
    """
    if mode == "status":
        check_status()
        return

    log = _setup_logging()
    log.info(f"{'=' * 50}")
    log.info(f"🏒 Snipe Tracker Runner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"   Mode: {mode}")
    log.info(f"{'=' * 50}")

    try:
        if mode in ("full", "grade"):
            step_grade_yesterday(log)

        pred_df = None
        report_path = None

        if mode in ("full", "predict"):
            pred_df, _, report_path = step_predict_today(log)

        if mode in ("full", "predict", "notify"):
            if mode == "notify":
                # Re-send the latest report
                from src.predictions.tracker import PICKS_FILE
                import pandas as pd
                if os.path.exists(PICKS_FILE):
                    ledger = pd.read_csv(PICKS_FILE)
                    today = datetime.now().strftime("%Y-%m-%d")
                    pred_df = ledger[ledger["prediction_date"] == today]
                report_path = _find_latest_report()

            step_notify(log, pred_df, report_path)

        log.info("✅ Done!")

    except Exception as e:
        log.error(f"❌ Runner failed: {e}", exc_info=True)
        raise


def main():
    """CLI entry point."""
    mode = "full"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lstrip("-")
        if arg in ("predict", "grade", "notify", "status"):
            mode = arg

    run(mode)


if __name__ == "__main__":
    main()
