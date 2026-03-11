"""
Snipe Tracker configuration.

Centralized settings for API endpoints, seasons, and model parameters.
"""

# NHL API (free, no key required)
NHL_API_BASE = "https://api-web.nhle.com/v1"
NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en"

# Default season — auto-detected from the current date.
# NHL seasons span two calendar years: a season starting in Oct 2025
# is '20252026'.  If today is before August we're still in the season
# that *started* last calendar year.

def _detect_season() -> str:
    from datetime import date
    today = date.today()
    start_year = today.year if today.month >= 8 else today.year - 1
    return f"{start_year}{start_year + 1}"

CURRENT_SEASON = _detect_season()

# Feature engineering defaults
ROLLING_WINDOW = 50       # games for rolling averages (full window)
RECENCY_WINDOW = 10       # most-recent sub-window inside the full window
RECENCY_WEIGHT = 1.5      # multiplier for recency window on volume metrics
MIN_GAMES_PLAYED = 5      # minimum games to include a player
STREAK_MIN_GAMES = 2      # minimum consecutive games to count as a "streak"

# Shot quality thresholds
HIGH_SHOT_RATE_THRESHOLD = 3.0  # shots per game above this = high volume shooter

# PDO regression detection
PDO_SELL_HIGH_THRESHOLD = 103   # PDO above this = unsustainable luck
SELL_HIGH_SCORING_PACE = 0.30   # goals/game minimum to qualify as "scoring high"

# Model defaults
TEST_SIZE = 0.2       # 20% of data held out for testing
RANDOM_STATE = 42     # reproducibility
