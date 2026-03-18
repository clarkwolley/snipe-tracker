# 🤖 Full Automation Setup — Zero Manual Input Needed

**Goal:** Everything runs automatically. You just get Telegram notifications. No commands, no logins, no manual steps.

---

## What Happens Automatically (Daily)

### Morning (9:00 AM)
```
1. runner.py --grade (from launchd/cron)
   ├─ Grade yesterday's predictions
   ├─ Grade yesterday's bets
   └─ Send Telegram summaries

2. runner.py --predict
   ├─ Generate today's predictions (~25 picks)
   ├─ AUTO-PLACE BETS on top picks
   ├─ Store bets in data/bets.csv
   └─ Send Telegram notifications:
      - "Placed 7 bets on BOS/NJD game"
      - Shows picks, probabilities, amounts wagered

3. runner.py --notify
   ├─ Send predictions via email
   └─ Send picks summary via Telegram
```

### Evening (Games finish)
```
(Automatic, no human input needed)
✓ Your bets are now in data/bets.csv
✓ Games are played
✓ Results are captured
✓ Waiting for next morning's auto-grade
```

### Next Morning (9:00 AM)
```
runner.py --grade (repeats)
├─ Fetches actual results
├─ Grades your bets: "3-1, +$145 profit"
├─ Compares vs model's picks
├─ Sends Telegram: "Record 3-1, ROI +23.5%"
└─ All stored in data/bets.csv
```

---

## Setup Steps (One-Time)

### Step 1: Configure Telegram (Already Done)
Your bot is already configured to:
- ✅ Receive `/bet` commands (if you want manual override)
- ✅ Send automatic notifications
- ✅ Show stats on demand

**Already in .env:**
```
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

### Step 2: Schedule the Automation

#### Option A: macOS (Using launchd)

You already have `com.snipetracker.daily.plist`. Just verify it's running:

```bash
# Check if loaded
launchctl list | grep snipetracker

# If not running, load it
launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist

# Check logs
tail -f ~/Library/Logs/snipetracker.log
```

**The plist runs:**
```
Daily at 9:00 AM:  python -m src.automation.runner
```

#### Option B: Linux/Windows (Using cron)

Add to crontab:
```bash
crontab -e

# Add this line:
0 9 * * * cd ~/Projects/snipe-tracker && python -m src.automation.runner
```

This runs every day at 9:00 AM.

#### Option C: systemd (Linux)

Create `/etc/systemd/system/snipetracker.service`:
```ini
[Unit]
Description=Snipe Tracker Daily Predictions and Betting
After=network.target

[Service]
Type=oneshot
User=youruser
WorkingDirectory=/home/youruser/Projects/snipe-tracker
ExecStart=/home/youruser/Projects/snipe-tracker/venv/bin/python -m src.automation.runner

[Install]
WantedBy=multi-user.target
```

Enable timer:
```bash
systemctl enable snipetracker.timer
systemctl start snipetracker.timer
```

---

## What You Get (Automatic Telegram Notifications)

### Morning Notification 1: Predictions
```
🏒 Snipe Tracker — 2026-03-20
25 players analyzed

🎯 Top Picks
 #  Player               Team    Prob    Streak
 1  Pastrnak             vs NJD   55%    
 2  Matthews             @LAK     54%     
 3  MacKinnon            vs MIN   52%     
...

📋 By Game
  NJD @ BOS: Pastrnak 55% / Pasta 43%
  LAK @ VGK: Matthews 54% / Stone 49%
```

### Morning Notification 2: Auto-Betting
```
🎲 AUTO-BETTING SUMMARY

Bets placed: 7
Total wagered: $380
Avg probability: 53.2%

Picks:
  • Pastrnak (BOS) $50 @ -110 (55%)
  • Matthews (LAK) $60 @ -120 (54%)
  • MacKinnon (COL) $50 @ -110 (52%)
  ...
```

### Next Morning Notification 1: Grade Predictions
```
📊 Scorecard — 2026-03-19

Players tracked: 25
Actually scored: 6
Predicted goals: 8
Hits: 5/8 (62% precision)

✅ Top Hits
  Pastrnak (BOS) — 55% → 2G
  Matthews (LAK) — 54% → 1G
  MacKinnon (COL) — 52% → 1G
```

### Next Morning Notification 2: Grade Bets
```
🎲 BET GRADING — 2026-03-19

Record: 5-2 (71%)
Wagered: $350
Profit: +$145.30
ROI: +41.5%

Details:
  ✅ Pastrnak (BOS) ($50) → +$45.45
  ✅ Matthews (LAK) ($60) → +$54.55
  ✅ MacKinnon (COL) ($50) → +$45.45
  ❌ Kucherov (TB) ($40) → -$40.00
  ...

vs Model:
  Your hit rate: 71%
  Model hit rate: 62%
  🔥 You beat the model!
```

---

## Key Files (Fully Automated)

```
src/automation/
├── runner.py                    ← Main orchestrator (runs daily)
├── auto_bettor.py              ← Auto-places bets
└── auto_grader.py              ← Auto-grades results

src/predictions/
├── bet_tracker.py              ← Stores bets in CSV
├── bet_grader.py               ← Grades bets
├── daily.py                    ← Generates predictions
└── report.py                   ← Creates HTML report

data/
├── bets.csv                    ← All your bets (auto-updated)
├── predictions.csv             ← All predictions (auto-updated)
└── graded.csv                  ← Graded results (auto-updated)

com.snipetracker.daily.plist    ← macOS scheduler
```

---

## Configuration Options

### Betting Limits (Edit `auto_bettor.py`)

```python
class BettingConfig:
    MIN_PROBABILITY = 0.50      # Only bet on 50%+ picks
    MAX_BETS_PER_DAY = 10       # Max 10 bets daily
    BASE_BET_AMOUNT = 50        # $50 base bet
    
    # Adjust bet size by confidence
    CONFIDENCE_TIERS = {
        0.50: 30,    # 50-52% → $30 (less confident)
        0.53: 50,    # 53-55% → $50 (normal)
        0.56: 75,    # 56-58% → $75 (more confident)
        0.59: 100,   # 59%+   → $100 (very confident)
    }
    
    MAX_SAME_TEAM = 3           # Max 3 bets from same team
    DEFAULT_ODDS = -110         # Standard player prop odds
```

To change:
```python
# Save with your custom config
from src.automation.auto_bettor import auto_place_bets, BettingConfig

config = BettingConfig()
config.MIN_PROBABILITY = 0.52  # Only 52%+
config.MAX_BETS_PER_DAY = 5    # Max 5/day
config.BASE_BET_AMOUNT = 100   # $100 base

placed_bets = auto_place_bets(predictions, config=config)
```

---

## Monitoring (Optional)

### View Logs
```bash
# macOS
log stream --predicate 'eventMessage contains "snipetracker"'

# Or tail the file
tail -f ~/Library/Logs/snipetracker.log
```

### Manual Overrides

If you **don't** want the auto-bet today, you can:

1. **Temporarily disable:**
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.snipetracker.daily.plist
   ```

2. **Place custom bets manually:**
   ```bash
   python bet_cli.py place "McDavid" 100 -120 2026-03-20
   ```

3. **Re-enable automation:**
   ```bash
   launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist
   ```

### Check Next Run
```bash
# macOS
launchctl list com.snipetracker.daily

# View the plist
cat ~/Library/LaunchAgents/com.snipetracker.daily.plist
```

---

## Data Flow (Automated)

```
Model Predictions (daily.py)
       ↓
auto_bettor.py (automatically select top picks)
       ↓
BetTracker (store in data/bets.csv)
       ↓
Telegram notification: "Placed 7 bets"
       ↓
  [Games play]
       ↓
bet_grader.py (fetch results, grade bets)
       ↓
BetTracker (update bets.csv with results)
       ↓
Telegram notification: "3-1 record, +$145 profit"
       ↓
Long-term tracking (compare vs model, ROI, calibration)
```

---

## What You Need to Do

**NOTHING!** It's fully automated. 

Just:
1. ✅ Set up scheduler (launchd/cron)
2. ✅ Let it run

You'll get:
- Daily predictions via Telegram
- Daily auto-placed bets (in CSV)
- Daily bet grading and ROI reports
- Long-term stats on demand

---

## Testing the Automation

### Dry Run (No Real Bets)
```bash
cd ~/Projects/snipe-tracker
python -m src.automation.runner --predict
```

This will:
- Generate predictions
- Show which bets WOULD be placed
- NOT store them in CSV
- Let you review first

### Manual One-Off Grade
```bash
python -m src.automation.runner --grade
```

This will:
- Grade yesterday's bets (if any)
- Send you the Telegram summary

### Check Status
```bash
python -m src.automation.runner --status
```

Shows:
- Email configured ✅
- Telegram configured ✅
- Model trained ✅
- Latest report
- Betting stats

---

## Troubleshooting

### Scheduler Not Running

**macOS:**
```bash
# Is it loaded?
launchctl list | grep snipetracker

# If not, load it
launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist

# Check for errors
log show --predicate 'processImagePath contains "python"' --last 1h
```

**Linux cron:**
```bash
# Check crontab
crontab -l

# Test manually
cd ~/Projects/snipe-tracker && python -m src.automation.runner
```

### No Telegram Notifications

```bash
# Check .env has token and chat ID
cat .env | grep TELEGRAM

# Test manually
python3 << 'EOF'
from src.notifications.telegram_sender import send_picks
from src.predictions.daily import predict_tonight

pred = predict_tonight()
send_picks(pred)
EOF
```

### Bets Not Being Placed

```bash
# Check MIN_PROBABILITY threshold
python3 << 'EOF'
from src.automation.auto_bettor import auto_place_bets
from src.predictions.daily import predict_tonight

pred = predict_tonight()
results = auto_place_bets(pred, dry_run=True)  # Don't actually place
print(results)
EOF
```

---

## Summary

✅ Predictions: Automated daily
✅ Betting: Automated (smart selection, proper sizing)
✅ Grading: Automated (after games)
✅ Notifications: Via Telegram
✅ Tracking: In CSV files
✅ Analysis: Built-in ROI, calibration, model comparison

**You just wake up and read your Telegram messages. Everything else is automatic.** 🤖

🐕 RiRi handles the rest!
