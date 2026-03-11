"""
Bulk data collector — efficiently pulls full seasons of game data.

💡 KEY CONCEPT: Instead of scanning day-by-day (slow!), we use team
schedules to discover ALL game IDs for a season in just 32 API calls.
Then we fetch boxscores only for games we don't already have.

This is the difference between O(days_in_season) and O(teams) for
discovery — a 10x reduction in API calls.

Usage:
    python -m src.data.collect_bulk              # Collect both seasons
    python -m src.data.collect_bulk --season 20232024  # Just one season
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd

from src.data import nhl_api
from src.data.collector import get_game_player_stats


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
GAME_LOG_FILE = os.path.join(DATA_DIR, "game_log.csv")
GAME_RESULTS_FILE = os.path.join(DATA_DIR, "game_results.csv")
CHECKPOINT_INTERVAL = 50  # save every N games


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_existing_game_ids() -> set:
    """Load game IDs we've already collected to avoid re-fetching."""
    if not os.path.exists(GAME_LOG_FILE):
        return set()
    df = pd.read_csv(GAME_LOG_FILE, usecols=["game_id"])
    return set(df["game_id"].unique())


def discover_season_games(season: str) -> pd.DataFrame:
    """
    Discover ALL games for a season by scanning team schedules.

    Makes 32 API calls (one per team) and deduplicates game IDs.
    Also extracts game results (scores) from the schedule data —
    no extra API calls needed for the game winner model!

    Args:
        season: Season string like '20232024' or '20242025'

    Returns:
        DataFrame with one row per unique completed game:
        - game_id, game_date, home_team, away_team
        - home_score, away_score, home_win, total_goals
    """
    print(f"\n🔍 Discovering games for {season[:4]}-{season[4:]} season...")

    # Get all 32 teams from standings
    standings = nhl_api.get_standings()
    teams = [t["teamAbbrev"]["default"] for t in standings.get("standings", [])]
    print(f"   Found {len(teams)} teams")

    all_games = {}
    for i, team in enumerate(teams, 1):
        try:
            data = nhl_api.get_team_schedule(team, season)
            games = data.get("games", [])

            for game in games:
                gid = game["id"]
                if gid in all_games:
                    continue  # already seen from the other team
                if game.get("gameState") != "OFF":
                    continue  # not completed
                if game.get("gameType", 0) != 2:
                    continue  # skip preseason (1), playoffs (3)

                home = game.get("homeTeam", {})
                away = game.get("awayTeam", {})
                home_score = home.get("score", 0)
                away_score = away.get("score", 0)

                all_games[gid] = {
                    "game_id": gid,
                    "game_date": game.get("gameDate", "")[:10],
                    "home_team": home.get("abbrev", ""),
                    "away_team": away.get("abbrev", ""),
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_win": int(home_score > away_score),
                    "total_goals": home_score + away_score,
                }

            print(f"   [{i:2d}/{len(teams)}] {team}: {len(games)} games in schedule")
            # _get() already throttles via MIN_REQUEST_GAP, but add courtesy gap
            time.sleep(0.3)

        except nhl_api.NHLApiError as e:
            print(f"   [{i:2d}/{len(teams)}] ⚠️  {team}: {e}")

    df = pd.DataFrame(all_games.values())
    if not df.empty:
        df = df.sort_values("game_date").reset_index(drop=True)
    print(f"   ✅ Found {len(df)} unique completed regular-season games")
    return df


def collect_boxscores(
    game_ids: list[int],
    delay: float = 0.4,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch player-level boxscore data for a list of game IDs.

    Features:
    - Progress bar with ETA
    - Checkpoint saves every CHECKPOINT_INTERVAL games
    - Graceful error handling (skips failed games)

    Args:
        game_ids: List of NHL game IDs to fetch
        delay: Seconds between API calls
        checkpoint_path: Path for checkpoint CSV saves

    Returns:
        DataFrame with player-game stats for all fetched games.
    """
    total = len(game_ids)
    if total == 0:
        print("   No games to fetch!")
        return pd.DataFrame()

    est_minutes = (total * delay) / 60
    print(f"\n📦 Fetching {total} boxscores (~{est_minutes:.1f} min at {delay}s/call)...")

    all_frames = []
    failed = []
    start_time = time.time()

    for i, gid in enumerate(game_ids, 1):
        try:
            df = get_game_player_stats(gid)
            all_frames.append(df)
        except Exception as e:
            failed.append(gid)
            if len(failed) <= 5:
                print(f"   ⚠️  Game {gid}: {e}")

        # Progress update every 25 games
        if i % 25 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"   [{i:4d}/{total}] "
                f"{i/total*100:5.1f}% | "
                f"{rate:.1f} games/sec | "
                f"~{remaining/60:.1f} min remaining"
            )

        # Checkpoint save
        if checkpoint_path and i % CHECKPOINT_INTERVAL == 0 and all_frames:
            _save_checkpoint(all_frames, checkpoint_path)

        time.sleep(delay)

    if failed:
        print(f"   ⚠️  Failed to fetch {len(failed)} games")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"   ✅ Collected {len(combined):,} player-game rows from {total - len(failed)} games")
    return combined


def _save_checkpoint(frames: list, path: str):
    """Save intermediate results to avoid losing progress."""
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)


def save_game_log(new_data: pd.DataFrame):
    """
    Merge new player-game data with existing game_log.csv.

    Deduplicates by (game_id, player_id) so re-runs are safe.
    """
    _ensure_data_dir()

    if os.path.exists(GAME_LOG_FILE):
        existing = pd.read_csv(GAME_LOG_FILE)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["game_id", "player_id"], keep="last"
        )
    else:
        combined = new_data

    combined = combined.sort_values("game_date").reset_index(drop=True)
    combined.to_csv(GAME_LOG_FILE, index=False)
    print(f"💾 Game log: {len(combined):,} rows ({combined['game_id'].nunique()} games)")


def save_game_results(results_df: pd.DataFrame):
    """
    Merge new game results with existing game_results.csv.

    Deduplicates by game_id so re-runs are safe.
    """
    _ensure_data_dir()

    if os.path.exists(GAME_RESULTS_FILE):
        existing = pd.read_csv(GAME_RESULTS_FILE)
        combined = pd.concat([existing, results_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    else:
        combined = results_df

    combined = combined.sort_values("game_date").reset_index(drop=True)
    combined.to_csv(GAME_RESULTS_FILE, index=False)
    print(f"💾 Game results: {len(combined):,} games")


def collect_season(season: str, delay: float = 0.4):
    """
    Full collection pipeline for one season.

    1. Discover all game IDs via team schedules
    2. Filter out already-collected games
    3. Fetch new boxscores with checkpointing
    4. Save game results (from schedule data — free!)
    5. Merge into existing data files
    """
    # Step 1: Discover games
    results_df = discover_season_games(season)
    if results_df.empty:
        print("No games found for this season!")
        return

    # Step 2: Save game results immediately (no extra API calls!)
    save_game_results(results_df)

    # Step 3: Filter out games we already have boxscores for
    existing_ids = _load_existing_game_ids()
    new_game_ids = [
        gid for gid in results_df["game_id"].tolist()
        if gid not in existing_ids
    ]
    print(f"\n   Already have: {len(existing_ids)} games")
    print(f"   New to fetch: {len(new_game_ids)} games")

    if not new_game_ids:
        print("   Nothing new to collect! 🎉")
        return

    # Step 4: Fetch boxscores
    checkpoint_path = os.path.join(DATA_DIR, f"_checkpoint_{season}.csv")
    new_data = collect_boxscores(new_game_ids, delay=delay, checkpoint_path=checkpoint_path)

    if new_data.empty:
        return

    # Step 5: Merge into game_log.csv
    save_game_log(new_data)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🧹 Cleaned up checkpoint file")


def collect_all(seasons: list[str] | None = None, delay: float = 0.4):
    """
    Collect data for multiple seasons.

    Args:
        seasons: List of season strings. Defaults to last 2 seasons.
        delay: Seconds between API calls.
    """
    if seasons is None:
        seasons = ["20232024", "20242025"]

    print("🏒 SNIPE TRACKER — Bulk Data Collection")
    print("=" * 50)
    print(f"   Seasons: {', '.join(s[:4]+'-'+s[4:] for s in seasons)}")
    print(f"   API delay: {delay}s per call")

    start = time.time()
    for season in seasons:
        collect_season(season, delay=delay)

    elapsed = time.time() - start
    print(f"\n🎉 Done! Total time: {elapsed/60:.1f} minutes")

    # Print final data summary
    if os.path.exists(GAME_LOG_FILE):
        gl = pd.read_csv(GAME_LOG_FILE)
        print(f"\n📊 Final dataset:")
        print(f"   Player-game rows: {len(gl):,}")
        print(f"   Unique games:     {gl['game_id'].nunique():,}")
        print(f"   Unique players:   {gl['player_id'].nunique():,}")
        print(f"   Date range:       {gl['game_date'].min()} → {gl['game_date'].max()}")


def main():
    """CLI entry point."""
    seasons = None
    delay = 0.4

    args = sys.argv[1:]
    if "--season" in args:
        idx = args.index("--season")
        if idx + 1 < len(args):
            seasons = [args[idx + 1]]
    if "--fast" in args:
        delay = 0.25
    if "--slow" in args:
        delay = 0.6

    collect_all(seasons=seasons, delay=delay)


if __name__ == "__main__":
    main()
