"""
Data collector — transforms raw NHL API responses into clean DataFrames.

💡 KEY CONCEPT: This is the "ETL" layer (Extract, Transform, Load).
The API client extracts raw data, this module transforms it into
tidy DataFrames that are easy to analyze and feed into models.

Why DataFrames? They're like spreadsheets in code — rows and columns
that you can filter, sort, group, and do math on. Pandas DataFrames
are the standard data structure in Python data science.
"""

from datetime import date

import pandas as pd
from src.data import nhl_api


def get_todays_games() -> pd.DataFrame:
    """
    Get today's scheduled games as a clean DataFrame.

    The NHL API /schedule/now endpoint returns an entire 7-day
    gameWeek. We filter to only the current date so early-morning
    runs do not accidentally pick up stale games from prior days.

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
    today_str = date.today().isoformat()  # 'YYYY-MM-DD'
    games = []

    for day in schedule.get("gameWeek", []):
        if day["date"] != today_str:
            continue
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
            # NOTE: pp_pct and pk_pct are NOT in the standings API.
            # They get merged from the stats API below.
        })

    df = pd.DataFrame(rows)

    # Merge real PP% and PK% from the stats API.
    # The standings endpoint doesn't include special teams data (!),
    # so we pull it from api.nhle.com/stats which has the goods.
    try:
        special_teams = _get_special_teams_lookup()
        df = df.merge(special_teams, on="team_name", how="left")
        df["pp_pct"] = df["pp_pct"].fillna(0.20)
        df["pk_pct"] = df["pk_pct"].fillna(0.80)
    except Exception as e:
        print(f"  ⚠️  Could not fetch special teams stats: {e}")
        df["pp_pct"] = 0.20
        df["pk_pct"] = 0.80

    return df


def _get_special_teams_lookup() -> pd.DataFrame:
    """
    Fetch real PP% and PK% from the NHL stats API.

    Returns a small DataFrame with columns: team_name, pp_pct, pk_pct
    that can be merged into standings by team_name.
    """
    data = nhl_api.get_team_stats_summary()
    rows = []
    for team in data.get("data", []):
        rows.append({
            "team_name": team["teamFullName"],
            "pp_pct": team.get("powerPlayPct", 0.20),
            "pk_pct": team.get("penaltyKillPct", 0.80),
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


def get_team_goalies(team_abbrev: str) -> pd.DataFrame:
    """
    Get season stats for all goalies on a team.

    Args:
        team_abbrev: Three-letter team code (e.g., 'COL')

    Returns:
        DataFrame with goalie season stats:
        - player_id, first_name, last_name
        - games_played, wins, losses, ot_losses
        - goals_against_avg, save_pct
        - shutouts, goals_against
        - team: Team abbreviation
    """
    data = nhl_api.get_team_roster_stats(team_abbrev)
    rows = []

    for goalie in data.get("goalies", []):
        rows.append({
            "player_id": goalie["playerId"],
            "first_name": goalie["firstName"]["default"],
            "last_name": goalie["lastName"]["default"],
            "games_played": goalie.get("gamesPlayed", 0),
            "wins": goalie.get("wins", 0),
            "losses": goalie.get("losses", 0),
            "ot_losses": goalie.get("otLosses", 0),
            "goals_against_avg": goalie.get("goalsAgainstAverage", 0.0),
            "save_pct": goalie.get("savePctg", 0.0),
            "shutouts": goalie.get("shutouts", 0),
            "goals_against": goalie.get("goalsAgainst", 0),
            "team": team_abbrev,
        })

    return pd.DataFrame(rows)


def get_team_recent_schedule(team_abbrev: str) -> pd.DataFrame:
    """
    Get a team's season schedule as a clean DataFrame.

    Returns:
        DataFrame with columns: game_id, date, opponent, is_home, game_state.
    """
    data = nhl_api.get_team_schedule(team_abbrev)
    rows = []

    for game in data.get("games", []):
        home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
        away_abbrev = game.get("awayTeam", {}).get("abbrev", "")
        is_home = home_abbrev == team_abbrev
        opponent = away_abbrev if is_home else home_abbrev

        rows.append({
            "game_id": game["id"],
            "date": game["gameDate"][:10],  # 'YYYY-MM-DD'
            "opponent": opponent,
            "is_home": is_home,
            "game_state": game.get("gameState", ""),
        })

    return pd.DataFrame(rows)


def get_back_to_back_status(team_abbrev: str, game_date: str) -> dict:
    """
    Check if a team is on a back-to-back and calculate days rest.

    Args:
        team_abbrev: Three-letter team code
        game_date: Date of the game in 'YYYY-MM-DD' format

    Returns:
        Dict with:
        - is_back_to_back (bool): True if team played yesterday
        - days_rest (int): Days since last completed game
    """
    from datetime import datetime, timedelta

    schedule = get_team_recent_schedule(team_abbrev)
    if schedule.empty:
        return {"is_back_to_back": False, "days_rest": 3}

    # Filter to completed games before game_date
    completed = schedule[
        (schedule["game_state"] == "OFF") & (schedule["date"] < game_date)
    ].sort_values("date", ascending=False)

    if completed.empty:
        return {"is_back_to_back": False, "days_rest": 3}

    last_game_date = datetime.strptime(completed.iloc[0]["date"], "%Y-%m-%d")
    current_date = datetime.strptime(game_date, "%Y-%m-%d")
    days_rest = (current_date - last_game_date).days

    return {
        "is_back_to_back": days_rest == 1,
        "days_rest": days_rest,
    }


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
