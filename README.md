# 🏒 Snipe Tracker

A hockey prediction model that forecasts **goal scorers** and **daily game winners** using real NHL data.

## Project Structure

```
snipe-tracker/
├── src/
│   ├── data/          # NHL API client & data fetching
│   ├── features/      # Feature engineering (turning stats → signals)
│   ├── models/        # ML model training & evaluation
│   └── predictions/   # Daily prediction pipeline
├── tests/             # Unit & integration tests
├── notebooks/         # Jupyter notebooks for exploration
├── requirements.txt   # Python dependencies
└── README.md
```

## How It Works

1. **Data Pipeline** — Pulls player, goalie & game stats from the free NHL API
2. **Feature Engineering** — Builds predictive features including:
   - Rolling averages (goals, shots, points, TOI)
   - 🔥 **Hot Streaks** — goal streak, point streak, drought detection
   - 🧊 **Goalie Matchups** — opposing goalie save%, GAA, quality score
   - ⚡ **Shot Quality** — shots per TOI minute, shooting % trend, volume flags
   - 🏒 **Special Teams** — team PP%, PK%, PP-vs-PK matchup differentials
   - 😴 **Back-to-Back** — fatigue flags, days rest, rest advantage
3. **Goal Scorer Model** — Logistic regression / gradient boosting to predict which players score on a given night
4. **Game Winner Model** — Predicts which team wins each matchup
5. **Daily Predictions** — CLI tool that outputs tonight's picks

## Getting Started

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run daily predictions (once models are trained)
python -m src.predictions.daily
```

## Key Concepts

- **Feature Engineering**: The art of turning raw data into useful model inputs
- **Logistic Regression**: A simple but powerful model for yes/no predictions
- **Gradient Boosting**: A more advanced model that combines many weak learners
- **Train/Test Split**: How we know if our model actually works vs. just memorizing data
- **Hot Streaks**: Momentum detection — players on goal/point streaks get a boost
- **Goalie Matchups**: Opposing goalie quality inversely affects scoring probability
- **Special Teams**: PP% and PK% matchups reveal hidden edges in game predictions
- **Back-to-Back Fatigue**: Teams on B2B lose ~4-5% more often — fatigue is real!

## Data Source

All data comes from the **free NHL API** — no API key required!
- Base URL: `https://api-web.nhle.com`

---

*Built with curiosity and a love for hockey.* 🏒
