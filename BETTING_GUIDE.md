# 🤑 Snipe Tracker — Betting System Guide

## Overview

Place bets, track them alongside your predictions, and analyze your ROI in real-time.

**Key Features:**
- ✅ Record bets with name, amount, odds
- ✅ Auto-calculate implied probability & payouts
- ✅ Grade bets against actual results
- ✅ Show ROI, win %, calibration
- ✅ Compare your bets vs. model predictions (do you trust your model?)

---

## Quick Start

### Option 1: Command Line (Easiest for Testing)

```bash
# Place a bet
python bet_cli.py place "David Pastrnak" 50 -110 2026-03-20

# With notes (optional)
python bet_cli.py place "David Pastrnak" 50 -110 2026-03-20 "model said 55%"

# List your bets for a date
python bet_cli.py list 2026-03-20

# Grade bets after games finish
python bet_cli.py grade 2026-03-20

# See your stats
python bet_cli.py stats 2026-03-20
```

### Option 2: Python Code (For Integration)

```python
from src.predictions.bet_tracker import BetTracker

tracker = BetTracker()

# Place a bet
bet = tracker.place_bet(
    player_name="David Pastrnak",
    team="BOS",
    opponent="NJD",
    bet_amount=50,
    odds=-110,
    game_date="2026-03-20",
    notes="Model confidence 55%"
)

# Later, grade the bet
tracker.grade_bet(bet_id=1, actual_goals=2)

# Check stats
stats = tracker.get_stats(game_date="2026-03-20")
print(f"ROI: {stats['roi']:+.1f}%")
```

### Option 3: Telegram Bot (Your Preferred Way)

Send a message to your bot:
```
/bet David Pastrnak 50 -110 2026-03-20
/bet "David Pastrnak" 50 -110 2026-03-20 "model said 55%"
/bets — show your open bets for today
/stats — show your overall statistics
```

---

## Data Storage

All bets stored in: `data/bets.csv`

Columns:
```
bet_id          — Unique ID for this bet
date_placed     — When you placed it (ISO timestamp)
player_name     — Player you bet on
team            — Player's team
opponent        — Opponent team
bet_amount      — $ wagered
odds            — American odds (-110, +150, etc.)
implied_probability  — What the odds imply
game_date       — Game date (YYYY-MM-DD)
status          — open, won, lost, push, cancelled
actual_goals    — How many the player scored
payout          — Total payout if won
profit          — Payout minus bet amount
notes           — Your notes
```

---

## Understanding the Metrics

### Odds & Payouts

**American Odds:**
- Negative (-110, -120) = Favorite
  - -110 means you bet $110 to win $100
  - Payout = bet_amount × (100 / 110)
  
- Positive (+150, +200) = Underdog
  - +150 means you bet $100 to win $150
  - Payout = bet_amount × (150 / 100)

**Implied Probability:**
- What the odds imply about the actual probability
- -110 → ~52% implied
- +150 → ~40% implied
- Use this to compare against your model's probability

### ROI (Return on Investment)

```
ROI = (Total Profit / Total Wagered) × 100

Example:
  Total wagered: $200
  Total profit: +$50
  ROI = (50 / 200) × 100 = +25%
```

**Benchmark:**
- 0% = Break even
- +10% = Very good (beating the line)
- -10% = You're losing money on average
- Track over 50+ bets for real signal

---

## Workflow

### Daily Routine

1. **Model runs predictions** (you get 25-30 top picks)
2. **You review** and place bets (maybe 5-10 picks you trust most)
   ```bash
   python bet_cli.py place "Pastrnak" 50 -110 2026-03-20
   ```
3. **Games play** 
4. **After games end**, grade your bets
   ```bash
   python bet_cli.py grade 2026-03-20
   ```
5. **Review performance**
   ```bash
   python bet_cli.py stats 2026-03-20
   ```

### Monthly Review

```bash
python bet_cli.py stats  # All-time stats
```

This shows:
- Total record (W-L)
- Win % (what fraction hit?)
- ROI (are you beating the line?)
- Total profit/loss

---

## Key Insights

### Case Study: 3/16/2026

You placed 4 bets:
```
#1  ✅ Pastrnak   $50  @ -110  (52% implied)  → WON   (scored 2G)
#2  ❌ Kopitar    $75  @ -120  (55% implied)  → LOST  (scored 0G)
#3  ✅ Carcone    $30  @ +150  (40% implied)  → WON   (scored 1G)
#4  ✅ Keller     $40  @ +130  (44% implied)  → WON   (scored 1G)

Record: 3-1 (75% hit rate!)
Profit: -$52.55
ROI: -26.9%
```

**What happened:**
- Model's top pick (Kopitar 59.7%) had **PDO issues** — high shooting % but low xG
- When he didn't score, you lost the biggest bet
- Your other bets were on lower-confidence picks who actually scored
- **Lesson:** Trust the model on medium confidence, not the outliers with extreme PDO

### Red Flags to Watch

1. **Betting on high PDO players**
   - If model says 55% but PDO is 108+, it's inflated
   - Consider reducing bet size or skipping

2. **Chasing losses**
   - After a -26.9% day, don't immediately bet 2x size to recover
   - Stick to a unit strategy

3. **Ignoring implied probability**
   - If model says 55% but market implies 52%, that's a VALUE play
   - If model says 55% but market implies 60%, SKIP

4. **Stacking same team too much**
   - Bets on Carcone + Keller (both UTA) are correlated
   - If UTA offense sucks that night, you lose both

---

## Integration with Predictions

### Compare Bets vs. Predictions

```bash
python bet_cli.py grade 2026-03-20
```

Shows:
```
🤔 PREDICTIONS vs. BETS

Your bets vs. model's top picks:

  1. 🎯 Anze Kopitar         59.7% → ❌  (You bet big, model was overconfident)
  2.    Jaroslav Chmelar     57.5% → ❌  (Model top pick, you didn't bet)
  3.    Valeri Nichushkin    56.2% → ❌  (Model top pick, you didn't bet)
  4. 🎯 David Pastrnak       55.1% → ✅  (You bet, it hit!)
  5.    Cale Makar           54.3% → ❌  (Model top pick, you didn't bet)

🎯 = You bet on this pick
```

**Questions this answers:**
- Did you bet on the model's top picks?
- If you diverged, were you right?
- Should you trust the model more/less?

---

## Advanced: Calibration Check

**The ultimate test:** Do your betting results match the model's probabilities?

Example:
- Model says 100 players have 55% goal probability
- Statistically, ~55 should score
- If only 30 scored, model is overconfident
- If 75 scored, model is underconfident

Track this with:
```python
from src.predictions.bet_tracker import BetTracker

tracker = BetTracker()
stats = tracker.get_stats()

# If you bet 100 times with similar confidence:
# stats['win_pct'] should match model calibration
```

---

## Future Enhancements

- [ ] Bankroll management (Kelly Criterion)
- [ ] Unit sizing based on confidence
- [ ] A/B testing: model picks vs. your picks
- [ ] Correlation detection (flag correlated bets)
- [ ] Dashboard with equity curve
- [ ] Telegram bot for live updates

---

## FAQ

**Q: Can I place multiple bets on the same player?**
A: Yes! Each bet gets its own ID.

**Q: What if a player is scratched?**
A: Mark it as "push" or "cancelled" (manual for now).

**Q: Can I edit a bet after placing?**
A: Not yet. Delete and re-place if needed (or add edit feature).

**Q: How long should I track before judging ROI?**
A: 50+ bets minimum. 100+ is better. 20+ bets can be noise.

---

**Remember:** The goal isn't to get 100% hit rate. It's to:
1. Beat the closing line (your model vs. market)
2. Maintain positive ROI over time
3. Calibrate your confidence to reality

Good luck! 🐕
