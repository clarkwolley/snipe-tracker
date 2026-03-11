"""
Tests for the notification and automation modules.

All external calls (SMTP, Telegram API) are mocked — no real
credentials needed to run these tests. 🐶
"""

import os
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

import pytest
import pandas as pd

from src.notifications import settings


# --- Settings tests ----------------------------------------------------------


class TestSettings:
    def test_load_from_env_file(self, tmp_path):
        """Should parse .env file correctly."""
        env_content = (
            "# Comment line\n"
            "SMTP_USER=test@example.com\n"
            "SMTP_PASSWORD=secret123\n"
            "EMAIL_RECIPIENT=recipient@example.com\n"
            "\n"
            "TELEGRAM_BOT_TOKEN=123:ABC\n"
            "TELEGRAM_CHAT_ID=99999\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        with patch.object(settings, "ENV_FILE", env_file):
            result = settings.load_settings()

        assert result["SMTP_USER"] == "test@example.com"
        assert result["SMTP_PASSWORD"] == "secret123"
        assert result["TELEGRAM_BOT_TOKEN"] == "123:ABC"
        assert result["TELEGRAM_CHAT_ID"] == "99999"

    def test_env_vars_override_file(self, tmp_path):
        """Environment variables should take precedence over .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("SMTP_USER=from_file@example.com\n")

        with patch.object(settings, "ENV_FILE", env_file):
            with patch.dict(os.environ, {"SMTP_USER": "from_env@example.com"}):
                result = settings.load_settings()

        assert result["SMTP_USER"] == "from_env@example.com"

    def test_missing_env_file(self, tmp_path):
        """Should return empty dict (plus any env vars) if .env doesn't exist."""
        fake_path = tmp_path / "nonexistent.env"

        with patch.object(settings, "ENV_FILE", fake_path):
            with patch.dict(os.environ, {}, clear=True):
                result = settings.load_settings()

        # No keys from file, and we cleared env vars
        assert result.get("SMTP_USER") is None

    def test_is_email_configured(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "SMTP_USER=a@b.com\nSMTP_PASSWORD=pass\nEMAIL_RECIPIENT=c@d.com\n"
        )
        with patch.object(settings, "ENV_FILE", env_file):
            assert settings.is_email_configured() is True

    def test_is_email_not_configured(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("SMTP_USER=a@b.com\n")  # missing password & recipient
        with patch.object(settings, "ENV_FILE", env_file):
            assert settings.is_email_configured() is False

    def test_is_telegram_configured(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TELEGRAM_BOT_TOKEN=123:ABC\nTELEGRAM_CHAT_ID=999\n")
        with patch.object(settings, "ENV_FILE", env_file):
            assert settings.is_telegram_configured() is True

    def test_strips_quotes_from_values(self, tmp_path):
        """Should strip surrounding quotes from .env values."""
        env_file = tmp_path / ".env"
        env_file.write_text("SMTP_PASSWORD='my secret'\n")
        with patch.object(settings, "ENV_FILE", env_file):
            result = settings.load_settings()
        assert result["SMTP_PASSWORD"] == "my secret"


# --- Email sender tests ------------------------------------------------------


class TestEmailSender:
    def test_send_report_success(self, tmp_path):
        """Should send email via SMTP on success."""
        from src.notifications import email_sender

        # Create a fake HTML file
        html_file = tmp_path / "test_report.html"
        html_file.write_text("<html><body>Test</body></html>")

        fake_settings = {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "sender@test.com",
            "SMTP_PASSWORD": "password",
            "EMAIL_RECIPIENT": "recipient@test.com",
        }

        with patch("src.notifications.email_sender.load_settings", return_value=fake_settings):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
                mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

                result = email_sender.send_report(str(html_file))

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("sender@test.com", "password")
        mock_server.send_message.assert_called_once()

    def test_send_report_missing_config(self):
        """Should return False when email isn't configured."""
        from src.notifications import email_sender

        with patch("src.notifications.email_sender.load_settings", return_value={}):
            result = email_sender.send_report("/fake/path.html")

        assert result is False

    def test_send_report_missing_file(self):
        """Should return False when report file doesn't exist."""
        from src.notifications import email_sender

        fake_settings = {
            "SMTP_USER": "a@b.com",
            "SMTP_PASSWORD": "pass",
            "EMAIL_RECIPIENT": "c@d.com",
        }

        with patch("src.notifications.email_sender.load_settings", return_value=fake_settings):
            result = email_sender.send_report("/nonexistent/report.html")

        assert result is False


# --- Telegram sender tests ---------------------------------------------------


class TestTelegramSender:
    @pytest.fixture()
    def sample_predictions(self):
        """Minimal prediction DataFrame for testing."""
        return pd.DataFrame([
            {
                "player_id": 1,
                "name": "Test Player",
                "team": "COL",
                "opponent": "TOR",
                "position": "C",
                "is_home": 1,
                "goal_probability": 0.75,
                "rolling_goals_avg": 0.5,
                "rolling_shots_avg": 3.2,
                "season_goals": 30,
                "season_gp": 60,
                "goal_streak": 3,
                "point_streak": 5,
                "drought": 0,
                "is_hot": 1,
            },
            {
                "player_id": 2,
                "name": "Other Guy",
                "team": "TOR",
                "opponent": "COL",
                "position": "L",
                "is_home": 0,
                "goal_probability": 0.65,
                "rolling_goals_avg": 0.3,
                "rolling_shots_avg": 2.5,
                "season_goals": 20,
                "season_gp": 60,
                "goal_streak": 0,
                "point_streak": 0,
                "drought": 7,
                "is_hot": 0,
            },
        ])

    def test_send_picks_success(self, sample_predictions):
        """Should POST to Telegram API on success."""
        from src.notifications import telegram_sender

        fake_settings = {
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_CHAT_ID": "99999",
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch("src.notifications.telegram_sender.load_settings", return_value=fake_settings):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                result = telegram_sender.send_picks(sample_predictions)

        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["chat_id"] == "99999"

    def test_send_picks_not_configured(self, sample_predictions):
        """Should return False when Telegram isn't configured."""
        from src.notifications import telegram_sender

        with patch("src.notifications.telegram_sender.load_settings", return_value={}):
            result = telegram_sender.send_picks(sample_predictions)

        assert result is False

    def test_format_picks_message(self, sample_predictions):
        """Message should contain player names and probabilities."""
        from src.notifications.telegram_sender import _format_picks_message

        message = _format_picks_message(sample_predictions)

        assert "Test Player" in message
        assert "Other Guy" in message
        assert "Snipe Tracker" in message
        assert "COL" in message

    def test_format_grade_message(self):
        """Grade message should show scorecard stats."""
        from src.notifications.telegram_sender import _format_grade_message

        graded = pd.DataFrame([
            {
                "prediction_date": "2026-03-04",
                "name": "Test Player",
                "team": "COL",
                "played": 1,
                "actual_scored": 1,
                "actual_goals": 1,
                "predicted_goal": 1,
                "hit": 1,
                "goal_probability": 0.75,
            },
        ])

        message = _format_grade_message(graded)

        assert "Scorecard" in message
        assert "2026-03-04" in message
        assert "Test Player" in message


# --- Runner tests ------------------------------------------------------------


class TestRunner:
    def test_check_status_runs_without_error(self):
        """Status check should work even with nothing configured."""
        from src.automation.runner import check_status
        # Just make sure it doesn't crash
        check_status()

    def test_find_latest_report(self, tmp_path):
        """Should find the most recent report file."""
        from src.automation import runner

        # Create fake report files
        (tmp_path / "picks_2026-03-04.html").write_text("old")
        (tmp_path / "picks_2026-03-05.html").write_text("new")

        with patch.object(runner, "REPORT_DIR", str(tmp_path)):
            result = runner._find_latest_report()

        assert result is not None
        assert "2026-03-05" in result
