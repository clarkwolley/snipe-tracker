"""
Snipe Tracker configuration.

Centralized settings for API endpoints, seasons, and model parameters.
"""

# NHL API (free, no key required)
NHL_API_BASE = "https://api-web.nhle.com/v1"
NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en"

# Default season to pull data for
CURRENT_SEASON = "20242025"

# Feature engineering defaults
ROLLING_WINDOW = 10  # games for rolling averages
MIN_GAMES_PLAYED = 5  # minimum games to include a player
STREAK_MIN_GAMES = 2  # minimum consecutive games to count as a "streak"

# Shot quality thresholds
HIGH_SHOT_RATE_THRESHOLD = 3.0  # shots per game above this = high volume shooter

# Model defaults
TEST_SIZE = 0.2       # 20% of data held out for testing
RANDOM_STATE = 42     # reproducibility
