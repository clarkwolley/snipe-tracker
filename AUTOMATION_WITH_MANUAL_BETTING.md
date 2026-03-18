# 🤖 Automated Predictions + Manual Telegram Betting

**Perfect setup!** Here's what happens automatically vs. manually:

---

## What's Automated

### ✅ Daily (9:00 AM)
```
runner.py runs automatically:

1. Grade yesterday's predictions
   └─> Shows calibration/accuracy
   └─> Sends Telegram summary

2. Grade yesterday's bets (if any placed)
   └─> Shows: wins/losses, ROI, profit
   └─> Compares your picks vs model's picks
   └─> Sends Telegram summary

3. Generate today's predictions (~25 picks)
   └─> Saves HTML report
   └─> Sends via Telegram & email
   └─> Logs to console

You do NOTHING. Everything runs automatically!
```

---

## What's Manual (Via Telegram)

### ✅ During the Day (Anytime)
```
You review predictions and send:

/bet Pastrnak 50 -110 2026-03-20
/bet Matthews 60 -120 2026-03-20
/bet MacKinnon 50 -110 2026-03-20

bot_server.py listens for these commands:

- Parses the command
- Validates format
- Stores in data/bets.csv
- Sends confirmation: "✅ Bet placed"

You control which picks you bet on.
The system just tracks them.
```

---

## Full Daily Flow

```
9:00 AM
├─ runner.py --grade (automated)
│  ├─ Grade yesterday's bets
│  └─ Telegram: "3-1 record, +$145 profit"
│
├─ runner.py --predict (automated)
│  ├─ Generate 25 predictions
│  └─ Telegram: "Top picks: Pastrnak 55%, Matthews 54%..."
│
└─ runner.py --notify (automated)
   ├─ Send email report
   └─ Done!

10:00 AM - 11:00 PM (YOU)
└─ Review predictions
   └─ Send /bet commands via Telegram
      ├─ /bet Pastrnak 50 -110 2026-03-20
      ├─ /bet Matthews 60 -120 2026-03-20
      └─ Confirmations saved to data/bets.csv

Next 9:00 AM
└─ runner.py --grade (automated)
   ├─ Grade your bets
   └─ Telegram: "Your bets 3-1, ROI +41%"
```

---

## One-Time Setup

### 1. Start bot_server.py (listens for /bet commands)

**Option A: Manual (for testing)**
```bash
# Terminal 1: Your daily automation
launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist

# Terminal 2: Bot listener (accepts /bet commands)
python bot_server.py --dev
```

**Option B: Ngrok (for real Telegram integration)**
```bash
# Terminal 1: Bot server
python bot_server.py --dev

# Terminal 2: Expose with ngrok
ngrok http 5000

# Terminal 3: Set webhook
python bot_server.py --set-webhook https://abc123.ngrok.io/telegram

# Now /bet commands work from Telegram!
```

### 2. Verify launchd scheduler is running

```bash
# macOS
launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist

# Linux: Add to crontab
crontab -e
# 0 9 * * * cd ~/Projects/snipe-tracker && python -m src.automation.runner
```

**That's it!** Two things running:
1. `runner.py` — Predictions & grading (automated daily)
2. `bot_server.py` — Listens for /bet commands (always running)

---

## What You Do Each Day

### Morning
```
Wake up, check Telegram:

📊 Predictions: "25 players analyzed, top pick Pastrnak 55%"

📊 Yesterday's Results: "3-1 record, +$145 profit, ROI +41%"

Done! Read the summaries.
```

### During the Day
```
Review the predictions.

Pick the ones you trust most and send:

/bet Pastrnak 50 -110 2026-03-20
/bet Matthews 60 -120 2026-03-20

Bot replies:
  ✅ Bet placed
  Pastrnak, $50 @ -110
  Bet ID: #47
  Stored in data/bets.csv

(You decide which ones to bet on!)
```

### Games Finish
```
Games play, results come in.
(Automatic — you do nothing)
```

### Next Morning
```
runner.py automatically grades your bets:

Telegram: "Your Record: 3-1, ROI +41%, Profit +$145"

Shows:
- Win-loss record
- Profit/loss
- Comparison: Your hits vs model's picks

All stored in data/bets.csv
```

---

## Telegram Commands

```
/bet Name Amount Odds Date [Notes]
  Place a bet
  /bet Pastrnak 50 -110 2026-03-20
  /bet "David Pastrnak" 50 -110 2026-03-20 "model said 55%"

/bets [Date]
  Show open bets
  /bets 2026-03-20

/stats
  Show your stats
  /stats

/grade Date
  Grade bets for a date (manual override)
  /grade 2026-03-20

/help
  Show commands
  /help
```

---

## Your Data (Auto-Updated)

### data/bets.csv
```
bet_id, date_placed, player_name, team, odds, game_date, status, actual_goals, profit
1, 2026-03-20T10:15, Pastrnak, BOS, -110, 2026-03-20, won, 2, +45.45
2, 2026-03-20T10:30, Matthews, LAK, -120, 2026-03-20, won, 1, +54.55
3, 2026-03-20T10:45, MacKinnon, COL, -110, 2026-03-20, lost, 0, -50.00
```

Updated automatically:
- When you place bets (bot_server.py)
- After games finish (runner.py --grade)

### data/predictions.csv
```
Automatically saved by runner.py

prediction_date, player_name, team, goal_probability, ...
2026-03-20, Pastrnak, BOS, 0.551, ...
2026-03-20, Matthews, LAK, 0.542, ...
```

---

## Monitoring

### Check if things are running

```bash
# Is launchd scheduler loaded?
launchctl list | grep snipetracker

# Is bot_server running?
ps aux | grep bot_server

# Check logs
tail -f ~/Library/Logs/snipetracker.log
```

### Manual Testing

```bash
# Test predictions
python -m src.automation.runner --predict

# Test grading
python -m src.automation.runner --grade

# Check status
python -m src.automation.runner --status

# Test bot locally
python bot_server.py --dev
# Then send /bet commands from Telegram
```

---

## What Each Component Does

| Component | What | When | Manual? |
|-----------|------|------|---------|
| runner.py | Grades predictions | Daily 9am | ❌ Auto |
| runner.py | Grades bets | Daily 9am | ❌ Auto |
| runner.py | Generates predictions | Daily 9am | ❌ Auto |
| runner.py | Sends Telegram summary | Daily 9am | ❌ Auto |
| bot_server.py | Listens for /bet | Always | ✅ Manual |
| You | Send /bet commands | During day | ✅ Manual |
| You | Review Telegram | Daily | ✅ Manual |

---

## Telegram Notifications You'll Get

### 9:00 AM — Predictions
```
🏒 Snipe Tracker — 2026-03-20
25 players analyzed

🎯 Top Picks
 #  Player            Prob  
 1  Pastrnak          55%
 2  Matthews          54%
 3  MacKinnon         52%
 ...

Full report: http://link-to-report.html
```

### 9:05 AM — Yesterday's Bets (if any)
```
🎲 BET GRADING — 2026-03-19

Record: 3-1 (75%)
Wagered: $200
Profit: +$85.50
ROI: +42.8%

✅ Pastrnak ($50) → +$45.45
❌ MacKinnon ($50) → -$50.00
✅ Matthews ($60) → +$54.55
✅ Keller ($40) → +$35.50

vs Model:
  Your hit rate: 75%
  Model hit rate: 62%
  🔥 You beat the model!
```

---

## Key Advantages

✅ **Predictions:** Fully automated (no manual input needed)
✅ **Grading:** Fully automated (after games)
✅ **Betting:** YOU control it (send /bet when you trust the pick)
✅ **Tracking:** Automatic (all bets stored in CSV)
✅ **Analysis:** Automatic (ROI, hit rate, model comparison)

You get:
- Automated daily predictions
- Manual control over betting
- Automatic grading and analysis
- Telegram notifications for everything

---

## Files You Need Running

### 1. Daily Automation (launchd/cron)
```bash
# macOS: Already in LaunchAgents
launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist

# Linux: Add to crontab
0 9 * * * cd ~/Projects/snipe-tracker && python -m src.automation.runner
```

### 2. Telegram Bot Listener (always)
```bash
# Option A: Local (for testing)
python bot_server.py --dev

# Option B: Production (with ngrok)
python bot_server.py --set-webhook https://yourdomain.com/telegram

# Option C: Cloud (Render, DigitalOcean)
Deploy bot_server.py to your cloud provider
```

---

## Summary

| Want | What You Get |
|------|--------------|
| Automated predictions | ✅ 9am daily |
| Manual betting | ✅ /bet commands |
| Automated grading | ✅ 9am next day |
| Telegram notifications | ✅ For everything |
| ROI tracking | ✅ In data/bets.csv |
| Model comparison | ✅ Auto-calculated |

**Perfect balance!** 🐕

Machine does the heavy lifting.
You decide which bets to place.
Everything gets tracked and analyzed.

---

## Next Steps

1. **Verify scheduler** (5 mins)
   ```bash
   launchctl load ~/Library/LaunchAgents/com.snipetracker.daily.plist
   ```

2. **Start bot server** (1 min)
   ```bash
   python bot_server.py --dev
   ```

3. **Test predictions** (5 mins)
   ```bash
   python -m src.automation.runner --predict
   ```

4. **Send test bet** (1 min)
   ```
   /bet Pastrnak 50 -110 2026-03-20
   ```

5. **Done!** Everything else is automatic ✅

See `TELEGRAM_BOT_SETUP.md` for bot setup details.

🐕 RiRi got you covered!
