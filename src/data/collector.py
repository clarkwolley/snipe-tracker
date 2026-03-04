"""
Data collector — transforms raw NHL API responses into clean DataFrames.

💡 KEY CONCEPT: This is the "ETL" layer (Extract, Transform, Load).
The API client extracts raw data, this module transforms it into
tidy DataFrames that are easy to analyze and feed into models.

Why DataFrames? They're like spreadsheets in code — rows and columns
that you can filter, sort, group, and do math on. Pandas DataFrames
are the standard data structure in Python data science.
"""

import pandas as pd
from src.data import nhl_api


def get_todays_games() -> pd.DataFrame:
    """
    Get today's scheduled games as a clean DataFrame.

    Returns:
        DataFrame with columns:
        - game_id: Unique game identifier
        - date: Game date
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - home_name: Home team full name
        - away_name: Away team full name
        - game_state: 'FUT' (future), 'LIVE', 'OFF' (final)
    """
    schedule = nhl_api.get_schedule()
    games = []

    for day in schedule.get("gameWeek", []):
        for game in day.get("games", []):
            games.append({
                "game_id": game["id"],
                "date": day["date"],
                "home_team": game["homeTeam"]["abbrev"],
                "away_team": game["awayTeam"]["abbrev"],
                "home_name": game["homeTeam"].get("placeName", {}).get("default", ""),
                "away_name": game["awayTeam"].get("placeName", {}).get("default", ""),
                "game_state": game["gameState"],
            })

    return pd.DataFrame(games)


def get_standings_df() -> pd.DataFrame:
    """
    Get current standings as a DataFrame.

    Returns:
        DataFrame with columns:
        - team: Team abbreviation (e.g., 'COL')
        - team_name: Full team name
        - games_played, wins, losses, ot_losses
        - points, point_pct: League points and percentage
        - goals_for, goals_against, goal_diff: Goal metrics
        - home_wins, home_losses, road_wins, road_losses
        - streak_code: Current streak (e.g., 'W3', 'L1')

    💡 CONCEPT: We pick specific columns that are useful as
    "features" for our model. Not everything the API returns
    matters — feature selection is about choosing signals
    and ignoring noise.
    """
    data = nhl_api.get_standings()
    rows = []

    for team in data.get("standings", []):
        rows.append({
            "team": team["teamAbbrev"]["default"],
            "team_name": team["teamName"]["default"],
            "games_played": team["gamesPlayed"],
            "wins": team["wins"],
            "losses": team["losses"],
            "ot_losses": team["otLosses"],
            "points": team["points"],
            "point_pct": team["pointPctg"],
            "goals_for": team["goalFor"],
            "goals_against": team["goalAgainst"],
            "goal_diff": team["goalDifferential"],
            "home_wins": team["homeWins"],
            "home_losses": team["homeLosses"],
            "road_wins": team["roadWins"],
            "road_losses": team["roadLosses"],
            "streak_code": team.get("streakCode", ""),
        })

    return pd.DataFrame(rows)


def get_team_skaters(team_abbrev: str) -> pd.DataFrame:
    """
    Get season stats for all skaters on a team.

    Args:
        team_abbrev: Three-letter team code (e.g., 'COL')

    Returns:
        DataFrame with columns:
        - player_id, first_name, last_name, position
        - games_played, goals, assists, points
        - shots, shooting_pct
        - pp_goals, sh_goals, gw_goals, ot_goals
        - plus_minus, pim
        - team: Team abbreviation (added for joining later)

    💡 CONCEPT: These per-player season stats become the basis
    for our features. A player who has 30 goals in 60 games
    scores at 0.5 goals/game — that rate is more useful than
    the raw total because it accounts for games played.
    """
    data = nhl_api.get_team_roster_stats(team_abbrev)
    rows = []

    for skater in data.get("skaters", []):
        rows.append({
            "player_id": skater["playerId"],
            "first_name": skater["firstName"]["default"],
            "last_name": skater["lastName"]["default"],
            "position": skater["positionCode"],
            "games_played": skater["gamesPlayed"],
            "goals": skater["goals"],
            "assists": skater["assists"],
            "points": skater["points"],
            "shots": skater["shots"],
            "shooting_pct": skater["shootingPctg"],
            "pp_goals": skater["powerPlayGoals"],
            "sh_goals": skater["shorthandedGoals"],
            "gw_goals": skater["gameWinningGoals"],
            "ot_goals": skater["overtimeGoals"],
            "plus_minus": skater["plusMinus"],
            "pim": skater["penaltyMinutes"],
            "team": team_abbrev,
        })

    return pd.DataFrame(rows)


def get_game_player_stats(game_id: int) -> pd.DataFrame:
    """
    Get per-player stats from a single game's boxscore.

    Args:
        game_id: NHL game ID

    Returns:
        DataFrame with one row per player per game:
        - game_id, player_id, name, position, team, is_home
        - goals, assists, points, shots, toi
        - hits, blocks, giveaways, takeaways
        - pp_goals, plus_minus, pim

    💡 CONCEPT: This is our most granular data — what each player
    did in each game. When we collect hundreds of these, we can
    build a dataset like:
    "Player X, playing at home, against team Y, scored 1 goal"
    That's exactly what we need to train a prediction model.
    """
    box = nhl_api.get_boxscore(game_id)
    rows = []

    game_date = box.get("gameDate", "")
    player_stats = box.get("playerByGameStats", {})

    for side, is_home in [("awayTeam", False), ("homeTeam", True)]:
        team_data = player_stats.get(side, {})
        team_abbrev = box.get(side, {}).get("abbrev", "")

        # Forwards and defense have the same stat structure
        for group in ["forwards", "defense"]:
            for player in team_data.get(group, []):
                rows.append({
                    "game_id": game_id,
                    "game_date": game_date,
                    "player_id": player["playerId"],
                    "name": player["name"]["default"],
                    "position": player["position"],
                    "team": team_abbrev,
                    "is_home": is_home,
                    "goals": player.get("goals", 0),
                    "assists": player.get("assists", 0),
                    "points": player.get("points", 0),
                    "shots": player.get("sog", 0),
                    "toi": player.get("toi", "0:00"),
                    "hits": player.get("hits", 0),
                    "blocks": player.get("blockedShots", 0),
                    "giveaways": player.get("giveaways", 0),
                    "takeaways": player.get("takeaways", 0),
                    "pp_goals": player.get("powerPlayGoals", 0),
                    "plus_minus": player.get("plusMinus", 0),
                    "pim": player.get("pim", 0),
                })

    return pd.DataFrame(rows)


def get_all_skaters() -> pd.DataFrame:
    """
    Get season stats for every skater across all 32 NHL teams.

    Returns:
        Combined DataFrame of all skaters from all teams.
        Same columns as get_team_skaters() output.

    ⚠️  Makes 32 API calls (one per team). Be patient!
    """
    standings = get_standings_df()
    all_teams = standings["team"].tolist()

    frames = []
    for team in all_teams:
        try:
            df = get_team_skaters(team)
            frames.append(df)
        except nhl_api.NHLApiError as e:
            print(f"  ⚠️  Failed to fetch {team}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
