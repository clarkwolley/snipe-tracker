"""Tests for the NHL injury data module."""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
import requests

from src.data.injuries import (
    fetch_injuries,
    get_team_injuries,
    get_unavailable_players,
    UNAVAILABLE_STATUSES,
    _ESPN_ID_TO_ABBREV,
)


# --- Fixtures ----------------------------------------------------------------

MOCK_ESPN_RESPONSE = {
    "injuries": [
        {
            "id": "17",  # Colorado Avalanche
            "displayName": "Colorado Avalanche",
            "injuries": [
                {
                    "status": "Out",
                    "athlete": {"displayName": "Gabriel Landeskog"},
                    "details": {"type": "Lower Body", "detail": "Surgery"},
                },
                {
                    "status": "Day-To-Day",
                    "athlete": {"displayName": "Valeri Nichushkin"},
                    "details": {"type": "Upper Body", "detail": "Soreness"},
                },
            ],
        },
        {
            "id": "6",  # Edmonton Oilers
            "displayName": "Edmonton Oilers",
            "injuries": [
                {
                    "status": "Injured Reserve",
                    "athlete": {"displayName": "Evander Kane"},
                    "details": {"type": "Wrist", "detail": "Surgery"},
                },
            ],
        },
    ],
}


@pytest.fixture
def mock_espn():
    """Mock the ESPN API response."""
    with patch("src.data.injuries.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_ESPN_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        yield mock_get


# --- Tests -------------------------------------------------------------------


def test_fetch_injuries_returns_dataframe(mock_espn):
    """fetch_injuries should return a DataFrame with expected columns."""
    df = fetch_injuries()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    expected_cols = {"player_name", "team", "status", "injury_type",
                     "injury_detail", "is_available"}
    assert expected_cols.issubset(set(df.columns))


def test_fetch_injuries_maps_teams_correctly(mock_espn):
    """ESPN team IDs should map to our abbreviations."""
    df = fetch_injuries()
    assert set(df["team"]) == {"COL", "EDM"}


def test_unavailable_status_flags(mock_espn):
    """Out and IR players should be marked unavailable; DTD should be available."""
    df = fetch_injuries()
    landeskog = df[df["player_name"] == "Gabriel Landeskog"].iloc[0]
    nichushkin = df[df["player_name"] == "Valeri Nichushkin"].iloc[0]
    kane = df[df["player_name"] == "Evander Kane"].iloc[0]

    assert landeskog["is_available"] == False  # Out
    assert nichushkin["is_available"] == True   # Day-To-Day = still might play
    assert kane["is_available"] == False        # Injured Reserve


def test_get_team_injuries(mock_espn):
    """Should filter injuries to a specific team."""
    col_injuries = get_team_injuries("COL")
    assert len(col_injuries) == 2
    assert all(col_injuries["team"] == "COL")


def test_get_unavailable_players(mock_espn):
    """Should return set of names for confirmed-out players."""
    unavailable = get_unavailable_players()
    assert "Gabriel Landeskog" in unavailable
    assert "Evander Kane" in unavailable
    assert "Valeri Nichushkin" not in unavailable  # DTD = still available


def test_get_unavailable_players_team_filter(mock_espn):
    """Team filter should narrow results."""
    unavailable = get_unavailable_players(teams=["COL"])
    assert "Gabriel Landeskog" in unavailable
    assert "Evander Kane" not in unavailable  # EDM player, not COL


def test_fetch_injuries_handles_api_failure():
    """Should return empty DataFrame on API failure, not crash."""
    with patch("src.data.injuries.requests.get", side_effect=requests.ConnectionError("timeout")):
        df = fetch_injuries()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


def test_espn_id_mapping_covers_all_nhl_teams():
    """Sanity check: we should map at least 32 ESPN team IDs."""
    assert len(_ESPN_ID_TO_ABBREV) >= 32


def test_unavailable_statuses_are_comprehensive():
    """The unavailable set should include the key exclusion statuses."""
    assert "Out" in UNAVAILABLE_STATUSES
    assert "Injured Reserve" in UNAVAILABLE_STATUSES
    assert "Suspension" in UNAVAILABLE_STATUSES
    # Day-To-Day should NOT be in this set
    assert "Day-To-Day" not in UNAVAILABLE_STATUSES
