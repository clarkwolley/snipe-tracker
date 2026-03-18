"""
COMPLETE SNIPE TRACKER SYSTEM — How Everything Flows Together
"""

YOUR COMPLETE SYSTEM:
═══════════════════════════════════════════════════════════════════════════════

                        DAILY PREDICTIONS
                              ↓
                    (Model runs, generates 25-30 picks)
                              ↓
                   runner.py → tracker.py (saves predictions)
                              ↓
                    ┌─────────────────────────┐
                    │ reports/predictions.html │  (you read this)
                    └─────────────────────────┘
                              ↓
                    You review in Telegram/browser
                              ↓
        ┌─────────────────────────────────────────────────┐
        │  YOU DECIDE: Which picks to bet on?             │
        │  Open Telegram, send:                            │
        │  /bet Pastrnak 50 -110 2026-03-20               │
        │  /bet McDavid 75 -120 2026-03-20                │
        └─────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────────┐
        │  bot_server.py (listens for /bet)        │
        │  telegram_webhook.py (parses command)    │
        │  telegram_bet_handler.py (validates)     │
        │  bet_tracker.py (saves to CSV)           │
        └──────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────┐
        │  data/bets.csv                        │
        │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
        │  Bet #1: Pastrnak, $50, -110, 3/20  │
        │  Bet #2: McDavid, $75, -120, 3/20   │
        │  ...                                  │
        └──────────────────────────────────────┘
                              ↓
                     GAMES PLAY
                              ↓
        Later that night/next day, you send:
        /grade 2026-03-20
                              ↓
        ┌─────────────────────────────────────────┐
        │  bet_grader.py (reconciles with results)│
        │  Matches bets to actual outcomes        │
        │  Calculates wins/losses/ROI             │
        │  Compares your picks vs model's picks   │
        └─────────────────────────────────────────┘
                              ↓
        Bot replies:
        ┌─────────────────────────────────────────┐
        │  Record: 2-0 on 3/20                    │
        │  Wagered: $125                          │
        │  Profit: +$42.50                        │
        │  ROI: +34.0%                            │
        │                                         │
        │  Comparison to Model:                   │
        │  You bet Pastrnak (55%) - HIT           │
        │  Model picked Kopitar (59.7%) - MISS   │
        │  You were smarter than the model!       │
        └─────────────────────────────────────────┘
                              ↓
                  Over time, gather data:
                  /stats (shows all-time record)
                              ↓
        ┌──────────────────────────────────────────┐
        │  Total Bets: 50                          │
        │  Record: 35-15 (70% hit rate)            │
        │  Wagered: $2,500                         │
        │  Profit: +$425                           │
        │  ROI: +17.0%                             │
        │                                          │
        │  Are you beating the market?             │
        │  Are you calibrated to your model?       │
        │  What patterns are working?              │
        └──────────────────────────────────────────┘
                              ↓
                   You now have:
          - Real data on your betting performance
          - Comparison point vs model's predictions
          - Framework for continuous improvement


═══════════════════════════════════════════════════════════════════════════════

FILE MAP & RESPONSIBILITIES:

PREDICTIONS (Your Model):
  src/predictions/player_features.py
    ├─ Loads player stats
    ├─ Detects PDO (via pdo_detector.py)
    ├─ Extracts powerplay context
    └─ Feeds to model
  
  src/models/goal_scorer.py
    ├─ XGBoost model (trained)
    └─ Predicts player goal probability
  
  src/predictions/tracker.py
    ├─ Saves daily predictions
    └─ Builds reports/predictions.html
  
  src/notifications/telegram_sender.py
    ├─ Sends daily predictions to you
    └─ Sends grade reports

SCORING/CALIBRATION:
  src/predictions/grader.py
    ├─ Reconciles predictions vs actual results
    ├─ Builds scorecard
    └─ Calculates calibration (Brier score)

BETTING SYSTEM:
  bot_server.py
    ├─ Flask app
    └─ Listens for incoming commands
  
  src/notifications/telegram_webhook.py
    ├─ Routes /bet, /stats, /grade, /help commands
    └─ Sends responses back to Telegram
  
  src/notifications/telegram_bet_handler.py
    ├─ Parses /bet command strings
    ├─ Validates formats
    └─ Returns confirmation messages
  
  src/predictions/bet_tracker.py
    ├─ Stores bets in data/bets.csv
    ├─ Calculates implied probability
    ├─ Calculates payouts (American odds)
    └─ Gets stats (wins/losses/ROI)
  
  src/predictions/bet_grader.py
    ├─ Reconciles bets with actual results
    ├─ Matches player names
    ├─ Calculates win/loss/ROI
    └─ Compares vs model's top picks

DATA STORAGE:
  data/bets.csv
    ├─ All your placed bets
    ├─ Status (open, won, lost)
    ├─ Profit/loss per bet
    └─ Notes (model confidence, etc)
  
  data/predictions.csv
    ├─ Daily predictions from model
    ├─ Player name, probability, team, opponent
    └─ Used for grading


═══════════════════════════════════════════════════════════════════════════════

DAILY CHECKLIST:

[ ] Morning:
    - Model runs automatically (launchd runner.py)
    - Predictions saved to reports/predictions.html
    - Telegram notification sent
    
[ ] Afternoon:
    - You review predictions
    - Decide which picks to bet on
    - Send /bet commands in Telegram
    - Confirmation saved automatically

[ ] Evening (after games):
    - Games finish
    - You send /grade Date command
    - Bot grades your bets
    - You see ROI, hit rate, model comparison
    - Bot logs results to data/bets.csv

[ ] Weekly:
    - Send /stats to see rolling performance
    - Review trends: what's working? what's not?
    - Adjust bet sizing, trust in model, etc.


═══════════════════════════════════════════════════════════════════════════════

KEY INTEGRATIONS:

1. TELEGRAM NOTIFICATIONS (2-way):
   - Predictions sent to you automatically (daily)
   - Grade summaries sent to you (after games)
   - YOU CAN REPLY with /bet commands
   - Bot processes immediately and confirms

2. MODEL FEEDBACK LOOP:
   - Your betting decisions tracked
   - Graded against actual results
   - Compared to model's top picks
   - Shows where model is right/wrong

3. HISTORICAL DATA:
   - data/bets.csv grows with each bet
   - data/predictions.csv tracks model output
   - Both can be used for analysis:
     * "What types of bets won?"
     * "When did I diverge from model?"
     * "Am I beating the closing line?"

4. CONTINUOUS IMPROVEMENT:
   - Every bet is a data point
   - Every grade is feedback
   - Over time, pattern emerges
   - Use patterns to improve predictions or betting


═══════════════════════════════════════════════════════════════════════════════

EXAMPLE: A COMPLETE DAY

MORNING (9am):
  [9:00] runner.py runs (automated)
  [9:05] Model generates 27 predictions for today's games
  [9:06] Predictions saved: reports/predictions.html
  [9:07] Telegram notification: "25 predictions ready for review"
  [9:15] You open Telegram, read predictions
  
YOUR DECISIONS (10am):
  Top 5 model picks:
    1. Pastrnak (55.1%) - You trust this
    2. Matthews (53.8%) - You're skeptical (low xG)
    3. MacKinnon (52.4%) - You like this
    4. McDavid (51.2%) - Could go either way
    5. Kucherov (50.8%) - You'll skip
  
  You decide to bet on: Pastrnak, MacKinnon
  
BETTING (10:15am):
  You send:
    /bet Pastrnak 50 -110 2026-03-20
    /bet MacKinnon 40 +120 2026-03-20
  
  Bot replies:
    Bet #1 Placed: Pastrnak $50 @ -110 (52.4% implied) - #1
    Bet #2 Placed: MacKinnon $40 @ +120 (45.5% implied) - #2
    Total wagered: $90
  
  Bets stored in data/bets.csv
  
GAMES PLAY (evening):
  Pastrnak: 1 goal - WIN!
  MacKinnon: 0 goals - LOSS
  
GRADING (next morning):
  You send:
    /grade 2026-03-20
  
  Bot replies:
    Record: 1-1
    Wagered: $90.00
    Profit: -$15.52
    ROI: -17.2%
    
    Model comparison:
    - Pastrnak (55.1%): You bet, HIT
    - Matthews (53.8%): Model pick, you skipped
    - MacKinnon (52.4%): You bet, MISS
  
STATS (week later):
  You send:
    /stats
  
  Bot replies:
    Total bets: 7
    Record: 5-2 (71.4%)
    Wagered: $650
    Profit: +$85
    ROI: +13.1%
    
  INSIGHT: You're beating the model! Your picks are better.


═══════════════════════════════════════════════════════════════════════════════

GETTING STARTED:

Step 1: Predictions Already Working
  ✅ Model runs daily
  ✅ Sends Telegram notification
  ✅ You review in browser or Telegram

Step 2: Add Betting (TODAY)
  [ ] Get bot token from @BotFather
  [ ] Add to .env: TELEGRAM_BOT_TOKEN=xxx
  [ ] Run: python bot_server.py --dev
  [ ] Test: /help
  [ ] Test bet: /bet Pastrnak 50 -110 2026-03-20

Step 3: Go Live (THIS WEEK)
  [ ] Install ngrok
  [ ] Set webhook
  [ ] Start using for real bets

Step 4: Track Performance (ONGOING)
  [ ] Every game day: /bet commands
  [ ] After games: /grade Date
  [ ] Weekly: /stats
  [ ] Monthly: Review patterns in data/bets.csv


═══════════════════════════════════════════════════════════════════════════════

YOU NOW HAVE:

✅ Predictions (daily, automated)
✅ Notifications (Telegram)
✅ Betting system (Telegram commands)
✅ Tracking (all bets stored)
✅ Grading (auto-matched with results)
✅ Analysis (ROI, calibration, comparisons)
✅ Feedback loop (model vs reality)

YOUR NEXT MOVE:

Wire up the bot server and start tracking real bets.
That's it. Everything else is automated.

See TELEGRAM_BOT_SETUP.md for detailed steps.

═══════════════════════════════════════════════════════════════════════════════
