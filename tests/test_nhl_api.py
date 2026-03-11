"""
Tests for the NHL API client's rate limiting and caching logic.

These tests use mocks — no actual API calls are made. 🐶
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.data import nhl_api


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset module-level state between tests."""
    nhl_api.clear_cache()
    nhl_api._last_request_time = 0.0
    yield
    nhl_api.clear_cache()


# --- Cache tests -------------------------------------------------------------


class TestCache:
    def test_cache_hit_skips_api_call(self):
        """Second call to same endpoint should come from cache."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"standings": []}
        fake_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=fake_response) as mock_get:
            result1 = nhl_api._get("/standings/now", cache_ttl=60)
            result2 = nhl_api._get("/standings/now", cache_ttl=60)

            assert result1 == result2
            assert mock_get.call_count == 1  # only ONE actual request

    def test_cache_miss_after_ttl_expires(self):
        """Cache should expire after TTL."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"data": "fresh"}
        fake_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=fake_response) as mock_get:
            nhl_api._get("/test", cache_ttl=0.1)
            time.sleep(0.15)  # wait for TTL to expire
            nhl_api._get("/test", cache_ttl=0.1)

            assert mock_get.call_count == 2  # both calls hit the API

    def test_cache_disabled_with_zero_ttl(self):
        """cache_ttl=0 should bypass cache entirely."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"data": "nocache"}
        fake_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=fake_response) as mock_get:
            nhl_api._get("/test", cache_ttl=0)
            nhl_api._get("/test", cache_ttl=0)

            assert mock_get.call_count == 2

    def test_clear_cache(self):
        """clear_cache() should wipe all cached entries."""
        nhl_api._cache_set("/foo", {"a": 1})
        assert nhl_api._cache_get("/foo") is not None

        nhl_api.clear_cache()
        assert nhl_api._cache_get("/foo") is None


# --- Retry / rate-limit tests ------------------------------------------------


class TestRetry:
    def test_retries_on_429_then_succeeds(self):
        """Should retry on 429 and eventually succeed."""
        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.headers = {}
        fail_response.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError("429 Too Many Requests")
        )

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"ok": True}
        ok_response.raise_for_status = MagicMock()

        with patch("requests.get", side_effect=[fail_response, ok_response]):
            with patch("time.sleep"):  # don't actually sleep in tests
                result = nhl_api._get("/test-retry", cache_ttl=0)

        assert result == {"ok": True}

    def test_respects_retry_after_header(self):
        """Should use Retry-After header value when present."""
        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.headers = {"Retry-After": "42"}
        fail_response.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError("429")
        )

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"ok": True}
        ok_response.raise_for_status = MagicMock()

        with patch("requests.get", side_effect=[fail_response, ok_response]):
            with patch("time.sleep") as mock_sleep:
                nhl_api._get("/test-header", cache_ttl=0)

        # Should have slept for 42 seconds (from Retry-After header)
        # Note: _throttle also calls time.sleep, so filter for the 42s call
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert 42.0 in sleep_args

    def test_raises_after_max_retries(self):
        """Should raise NHLApiError after exhausting all retries."""
        fail_response = MagicMock()
        fail_response.status_code = 429
        fail_response.headers = {}
        fail_response.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError("429")
        )

        with patch("requests.get", return_value=fail_response):
            with patch("time.sleep"):
                with pytest.raises(nhl_api.NHLApiError):
                    nhl_api._get("/doomed", cache_ttl=0)

    def test_non_429_error_raises_immediately(self):
        """Non-429 HTTP errors should raise without retrying."""
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.headers = {}
        fail_response.raise_for_status.side_effect = (
            __import__("requests").exceptions.HTTPError("500 Server Error")
        )

        with patch("requests.get", return_value=fail_response) as mock_get:
            with pytest.raises(nhl_api.NHLApiError):
                nhl_api._get("/server-error", cache_ttl=0)

        assert mock_get.call_count == 1  # no retries on 500


# --- Throttle tests ----------------------------------------------------------


class TestThrottle:
    def test_throttle_enforces_minimum_gap(self):
        """Rapid calls should be slowed by the throttle."""
        with patch("time.sleep") as mock_sleep:
            nhl_api._last_request_time = time.monotonic()  # "just called"
            nhl_api._throttle()

            # Should have slept to enforce the gap
            assert mock_sleep.called


# --- get_todays_games date filter tests --------------------------------------


class TestGetTodaysGames:
    """Verify get_todays_games() filters the 7-day gameWeek to today only."""

    FAKE_GAME_WEEK = {
        "gameWeek": [
            {
                "date": "2025-05-24",
                "games": [
                    {
                        "id": 2024021000,
                        "homeTeam": {"abbrev": "NYR", "placeName": {"default": "New York"}},
                        "awayTeam": {"abbrev": "CAR", "placeName": {"default": "Carolina"}},
                        "gameState": "OFF",
                    }
                ],
            },
            {
                "date": "2025-05-25",
                "games": [
                    {
                        "id": 2024021001,
                        "homeTeam": {"abbrev": "FLA", "placeName": {"default": "Florida"}},
                        "awayTeam": {"abbrev": "EDM", "placeName": {"default": "Edmonton"}},
                        "gameState": "FUT",
                    }
                ],
            },
            {
                "date": "2025-05-26",
                "games": [],
            },
        ]
    }

    @patch("src.data.nhl_api.get_schedule")
    @patch("src.data.collector.date")
    def test_returns_only_todays_games(self, mock_date, mock_schedule):
        """Should return only games matching today's date, not the whole week."""
        from src.data.collector import get_todays_games

        mock_schedule.return_value = self.FAKE_GAME_WEEK
        mock_date.today.return_value.isoformat.return_value = "2025-05-25"

        result = get_todays_games()

        assert len(result) == 1
        assert result.iloc[0]["home_team"] == "FLA"
        assert result.iloc[0]["away_team"] == "EDM"
        assert result.iloc[0]["date"] == "2025-05-25"

    @patch("src.data.nhl_api.get_schedule")
    @patch("src.data.collector.date")
    def test_excludes_yesterdays_games(self, mock_date, mock_schedule):
        """Yesterday's games (the old bug) should NOT appear."""
        from src.data.collector import get_todays_games

        mock_schedule.return_value = self.FAKE_GAME_WEEK
        mock_date.today.return_value.isoformat.return_value = "2025-05-25"

        result = get_todays_games()

        yesterday_games = result[result["date"] == "2025-05-24"]
        assert len(yesterday_games) == 0

    @patch("src.data.nhl_api.get_schedule")
    @patch("src.data.collector.date")
    def test_returns_empty_when_no_games_today(self, mock_date, mock_schedule):
        """Should return empty DataFrame when today has no games."""
        from src.data.collector import get_todays_games

        mock_schedule.return_value = self.FAKE_GAME_WEEK
        mock_date.today.return_value.isoformat.return_value = "2025-05-26"

        result = get_todays_games()

        assert result.empty
