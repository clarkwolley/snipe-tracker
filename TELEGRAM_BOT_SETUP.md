# 🤖 Telegram Bot Setup — Complete Guide

## What You're Building

A Telegram bot that lets you send `/bet Pastrnak 50 -110 2026-03-20` and it:
1. ✅ Parses the command
2. ✅ Places the bet
3. ✅ Sends confirmation back
4. ✅ Stores it for tracking

Just like your TODO system, but for bets!

---

## Prerequisites

### What You Need
- ✅ A Telegram account
- ✅ A Telegram bot (from @BotFather)
- ✅ Your bot's TOKEN
- ✅ A way to receive webhook updates (see options below)

### Step 1: Create Your Bot with @BotFather

1. Open Telegram
2. Search for `@BotFather`
3. Start a conversation
4. Send: `/newbot`
5. Follow prompts:
   - "What should your bot be called?" → `SnipeTracker` (or whatever)
   - "Give your bot a username" → `SnipeTrackerBot` (must be unique, must end in 'bot')
6. **Copy the token** → `123456:ABCdefGHIjklmnoPQRstuvWxyz`

### Step 2: Save Token to .env

```bash
# Edit .env in snipe-tracker root
TELEGRAM_BOT_TOKEN=123456:ABCdefGHIjklmnoPQRstuvWxyz
TELEGRAM_CHAT_ID=your-user-id-here
```

**To get your CHAT_ID:**
1. Message your bot anything
2. Visit: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
3. Look for `"id": 123456789` in the chat object
4. That's your CHAT_ID

---

## Option A: Local Development (Testing Only)

**This works offline, but Telegram can't reach you for webhooks.**

### Run in Dev Mode

```bash
cd ~/Projects/snipe-tracker
source venv/bin/activate

python bot_server.py --dev
```

You'll see:
```
📍 Running in DEVELOPMENT mode (localhost only)
📋 Available commands (send to bot):
   /bet Name Amount Odds Date [Notes]
   ...
```

### Test with Local Commands

```bash
# In another terminal, test the command parser:
source venv/bin/activate
python3 << 'EOF'
from src.notifications.telegram_bet_handler import parse_bet_command

text = "/bet Pastrnak 50 -110 2026-03-20 model said 55%"
parsed = parse_bet_command(text)
print(parsed)
EOF
```

✅ **Use this for testing locally**
❌ **Won't receive actual Telegram messages**

---

## Option B: Production (Real Bot) — 3 Ways

### Setup Overview

You need to tell Telegram where to send updates. Three options:

```
Telegram → Webhook URL (your server)
```

---

### Option B1: Ngrok (Easiest, Temporary)

**Perfect for testing on your home machine.**

1. **Install ngrok** (one-time)
   ```bash
   # macOS
   brew install ngrok/ngrok/ngrok
   
   # Or download from https://ngrok.com/download
   ```

2. **Start bot server in one terminal**
   ```bash
   cd ~/Projects/snipe-tracker
   source venv/bin/activate
   python bot_server.py --dev
   ```

3. **In another terminal, expose with ngrok**
   ```bash
   ngrok http 5000
   ```
   
   You'll see:
   ```
   Forwarding                    https://abc123.ngrok.io -> http://localhost:5000
   ```

4. **Set webhook (third terminal)**
   ```bash
   source venv/bin/activate
   python bot_server.py --set-webhook https://abc123.ngrok.io/telegram
   ```
   
   Response:
   ```
   ✅ Webhook set to: https://abc123.ngrok.io/telegram
   ```

5. **Now test from Telegram!**
   ```
   /bet Pastrnak 50 -110 2026-03-20
   ```

   ✅ Bot should reply instantly!

**Pros:**
- Easy to set up
- Works from home
- Free tier available

**Cons:**
- URL changes every time you restart (pay $5/month for static)
- Only works while ngrok is running
- Not suitable for 24/7 production

---

### Option B2: Cloud Server (Heroku, Render, DigitalOcean)

**For real 24/7 production.**

#### Using Render (Free Tier Available)

1. **Create Render account** at https://render.com

2. **Deploy from Git**
   - Click "New +" → "Web Service"
   - Connect your snipe-tracker GitHub repo
   - Build command: `pip install -r requirements.txt`
   - Start command: `python bot_server.py --webhook $WEBHOOK_URL`
   - Add environment variable:
     ```
     TELEGRAM_BOT_TOKEN = your_token_here
     WEBHOOK_URL = https://your-app-name.onrender.com/telegram
     ```

3. **Render gives you a URL** → `https://your-app-name.onrender.com`

4. **Bot automatically sets webhook on startup**

**Pros:**
- Free tier available
- Always running
- Real production URL
- Easy to deploy

**Cons:**
- Requires Git setup
- Slight startup delay

---

### Option B3: Your Own Server (VPS)

**If you have a VPS (DigitalOcean, AWS, etc.):**

1. **SSH into your server**
   ```bash
   ssh user@your-server-ip
   ```

2. **Clone snipe-tracker repo**
   ```bash
   git clone https://github.com/yourname/snipe-tracker
   cd snipe-tracker
   ```

3. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Set .env with your token**
   ```bash
   echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env
   echo "TELEGRAM_CHAT_ID=your_id_here" >> .env
   ```

5. **Run with gunicorn (production WSGI server)**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 "bot_server:create_app()" &
   ```

6. **Set webhook to your server URL**
   ```bash
   python bot_server.py --set-webhook https://your-server.com/telegram
   ```

7. **(Optional) Set up reverse proxy with nginx** for HTTPS

**Pros:**
- Full control
- Always running
- Can integrate with other services

**Cons:**
- Requires server management
- Need to handle HTTPS/SSL

---

## Testing Your Setup

### Test 1: Health Check
```bash
curl http://localhost:5000/
# Should return: {"status": "ok", "message": "Snipe Tracker Bot running"}
```

### Test 2: Status Check
```bash
curl http://localhost:5000/status
# Should show: telegram_configured: true
```

### Test 3: Send Real Command (from Telegram)
Open your bot in Telegram and send:
```
/help
```

Bot should reply with list of commands.

### Test 4: Place a Bet
```
/bet Pastrnak 50 -110 2026-03-20 test bet
```

Bot should reply:
```
✅ Bet Placed

Player: Pastrnak
Amount: $50.00
Odds: -110
Implied: 52.4%
Game: 2026-03-20
Bet ID: #1
Notes: test bet
```

### Test 5: List Your Bets
```
/bets 2026-03-20
```

Bot shows all bets for that date.

### Test 6: Grade Your Bets
```
/grade 2026-03-20
```

Bot grades and shows results!

---

## Full Command Reference

### Placing Bets
```
/bet Name Amount Odds GameDate [Notes]

Examples:
/bet Pastrnak 50 -110 2026-03-20
/bet "David Pastrnak" 50 -110 2026-03-20 "model said 55%"
/bet McDavid 75 -120 2026-03-20
```

### Viewing Bets
```
/bets [Date]

Examples:
/bets                    # All open bets
/bets 2026-03-20         # Just today's bets
```

### Statistics
```
/stats

Shows:
- Total bets placed
- Graded bets
- Win-loss record
- ROI
- Profit/loss
```

### Grading Results
```
/grade Date

Examples:
/grade 2026-03-20

Shows:
- Wins/losses
- Money wagered
- Profit/loss
- ROI
- Your picks vs. model's top picks
```

### Help
```
/help

Shows all available commands
```

---

## Troubleshooting

### "❌ TELEGRAM_BOT_TOKEN not configured"
**Fix:** Add to `.env`:
```
TELEGRAM_BOT_TOKEN=your_token_here
```

### Bot doesn't respond
**Check:**
1. Is bot_server.py running?
   ```bash
   ps aux | grep bot_server
   ```

2. Is webhook set?
   ```bash
   curl https://api.telegram.org/botYOUR_TOKEN/getWebhookInfo
   ```
   Should show: `"url": "https://your-url/telegram"`

3. Check bot_server logs for errors
   ```bash
   # If running in terminal, scroll up to see errors
   # Or check logs directory
   ls logs/
   ```

### "Webhook setup failed"
**Check:**
1. Is your URL accessible from internet?
   ```bash
   curl https://your-url/
   ```

2. Is it HTTPS (Telegram requires it)?
   - Ngrok: Auto-HTTPS ✅
   - Render: Auto-HTTPS ✅
   - Your server: Use Let's Encrypt (certbot)

3. Is port correct?
   - If running on port 5000 locally, ngrok is :5000
   - But production usually uses :443 for HTTPS

### Bot sends wrong reply
**Check:**
1. Is command parsed correctly?
   ```bash
   python3 -c "from src.notifications.telegram_bet_handler import parse_bet_command; print(parse_bet_command('/bet Pastrnak 50 -110 2026-03-20'))"
   ```

2. Check bot_server.py logs (prints JSON of updates received)

---

## Recommended Setup

**For you (development + testing):**

1. **Start with ngrok** (Option B1)
   - Quick to test
   - No server setup
   - Just do it locally

2. **When ready for production:**
   - Move to Render (Option B2) or your VPS
   - Set permanent webhook
   - Done!

---

## Next Steps

### Right Now (5 mins)
1. ✅ Get bot token from @BotFather
2. ✅ Add to .env
3. ✅ Get your CHAT_ID
4. ✅ Test with `python bot_server.py --dev`

### Today (15 mins)
1. ✅ Install ngrok
2. ✅ Run `ngrok http 5000`
3. ✅ Set webhook: `python bot_server.py --set-webhook <ngrok-url>/telegram`
4. ✅ Test from Telegram: `/bet Pastrnak 50 -110 2026-03-20`

### This Week
1. Deploy to Render or VPS for 24/7
2. Start using in real games
3. Track ROI over 50+ bets

---

## Files Used

```
bot_server.py                      ← Main Flask app (run this)
src/notifications/
  ├── telegram_webhook.py          ← Handles /bet commands
  ├── telegram_bet_handler.py       ← Parses command + creates bet
  └── settings.py                   ← Loads .env
src/predictions/
  ├── bet_tracker.py               ← Stores bets
  └── bet_grader.py                ← Grades results
```

---

## One More Thing

**Your bot is now part of your prediction system:**

```
Daily Predictions → You Review → /bet command → Tracked
                                      ↓
                              Bet Tracker CSV
                                      ↓
                         /grade command
                                      ↓
                    Compare: Model vs. Your Decisions
```

Pretty cool, right? 🐕

Go make some money!
