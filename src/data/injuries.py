"""
Injury report data from ESPN.

Fetches the current NHL injury report via ESPN's free API and returns
a clean DataFrame. ESPN is ideal here because:
- Free, no API key
- Updated frequently (multiple times per day)
- Consistent format across all major sports

Statuses we care about:
- "Out"            → definitely not playing → EXCLUDE from predictions
- "Injured Reserve"→ definitely not playing → EXCLUDE from predictions
- "Day-To-Day"     → might play            → FLAG in predictions
- "Suspension"     → not playing           → EXCLUDE from predictions
"""

import requests
import pandas as pd

ESPN_NHL_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
)

# ESPN team ID → our abbreviation (NHL API format).
# ESPN uses slightly different abbreviations for some teams.
_ESPN_ID_TO_ABBREV = {
    "25": "ANA", "1": "BOS", "2": "BUF", "3": "CGY", "7": "CAR",
    "29": "CBJ", "4": "CHI", "17": "COL", "9": "DAL", "5": "DET",
    "6": "EDM", "26": "FLA", "8": "LAK", "30": "MIN", "10": "MTL",
    "11": "NJD", "27": "NSH", "12": "NYI", "13": "NYR", "14": "OTT",
    "15": "PHI", "16": "PIT", "124292": "SEA", "18": "SJS", "19": "STL",
    "20": "TBL", "21": "TOR", "129764": "UTA", "22": "VAN", "37": "VGK",
    "28": "WPG", "23": "WSH",
}

# Statuses that mean the player is NOT available
UNAVAILABLE_STATUSES = {"Out", "Injured Reserve", "Suspension"}


def fetch_injuries() -> pd.DataFrame:
    """
    Fetch the current NHL injury report from ESPN.

    Returns:
        DataFrame with columns:
        - player_name: str (e.g., "Nathan MacKinnon")
        - team: str (our abbreviation, e.g., "COL")
        - status: str (e.g., "Out", "Day-To-Day")
        - injury_type: str (e.g., "Upper Body", "Lower Body")
        - injury_detail: str (e.g., "Strain", "Soreness")
        - is_available: bool (False if Out/IR/Suspended)

    Returns empty DataFrame on API failure (predictions still work,
    just without injury filtering — graceful degradation).
    """
    try:
        resp = requests.get(ESPN_NHL_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  ⚠️  Could not fetch injury data: {e}")
        return pd.DataFrame()

    rows = []
    for team_entry in data.get("injuries", []):
        espn_id = str(team_entry.get("id", ""))
        team_abbrev = _ESPN_ID_TO_ABBREV.get(espn_id, "")

        if not team_abbrev:
            continue

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            details = inj.get("details", {})
            status = inj.get("status", "Unknown")

            rows.append({
                "player_name": athlete.get("displayName", "Unknown"),
                "team": team_abbrev,
                "status": status,
                "injury_type": details.get("type", inj.get("longComment", "")),
                "injury_detail": details.get("detail", ""),
                "is_available": status not in UNAVAILABLE_STATUSES,
            })

    return pd.DataFrame(rows)


def get_team_injuries(team: str) -> pd.DataFrame:
    """Get injury report for a specific team."""
    all_injuries = fetch_injuries()
    if all_injuries.empty:
        return all_injuries
    return all_injuries[all_injuries["team"] == team].reset_index(drop=True)


def get_unavailable_players(teams: list[str] | None = None) -> set[str]:
    """
    Get set of player names confirmed OUT for today.

    Args:
        teams: Optional list of team abbreviations to filter.
               If None, returns all unavailable players league-wide.

    Returns:
        Set of player display names (e.g., {"Nathan MacKinnon", "Troy Terry"}).
    """
    injuries = fetch_injuries()
    if injuries.empty:
        return set()

    unavailable = injuries[~injuries["is_available"]]
    if teams:
        unavailable = unavailable[unavailable["team"].isin(teams)]

    return set(unavailable["player_name"])


def print_injury_report(teams: list[str] | None = None) -> pd.DataFrame:
    """
    Print a formatted injury report and return the full DataFrame.

    Args:
        teams: Optional list of team abbreviations to filter.
    """
    injuries = fetch_injuries()
    if injuries.empty:
        print("  ℹ️  No injury data available")
        return injuries

    if teams:
        injuries = injuries[injuries["team"].isin(teams)]

    if injuries.empty:
        print("  ✅ No injuries reported for today's teams")
        return injuries

    out = injuries[~injuries["is_available"]]
    dtd = injuries[injuries["is_available"] & (injuries["status"] == "Day-To-Day")]

    if not out.empty:
        print(f"\n  🚫 OUT ({len(out)} players):")
        for _, row in out.iterrows():
            reason = row["injury_type"]
            if row["injury_detail"]:
                reason += f" ({row['injury_detail']})"
            print(f"     {row['team']:>3} {row['player_name']:<25} {row['status']:<18} {reason}")

    if not dtd.empty:
        print(f"\n  ⚠️  DAY-TO-DAY ({len(dtd)} players):")
        for _, row in dtd.iterrows():
            reason = row["injury_type"]
            if row["injury_detail"]:
                reason += f" ({row['injury_detail']})"
            print(f"     {row['team']:>3} {row['player_name']:<25} {reason}")

    return injuries
