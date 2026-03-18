# ⚡ Telegram Bot — Quick Start (5 Minutes)

## Step 1: Get Your Bot Token (2 mins)

Open Telegram → Search `@BotFather` → Send `/newbot` → Get token:
```
123456:ABCdefGHIjklmnoPQRstuvWxyz
```

## Step 2: Save to .env (1 min)

```bash
cd ~/Projects/snipe-tracker
echo "TELEGRAM_BOT_TOKEN=123456:ABCdefGHIjklmnoPQRstuvWxyz" > .env
```

Get your chat ID by messaging your bot, then:
```bash
curl "https://api.telegram.org/bot123456:ABCdef/getUpdates"
# Find "id": 987654321 → that's your CHAT_ID
echo "TELEGRAM_CHAT_ID=987654321" >> .env
```

## Step 3: Test Locally (2 mins)

```bash
source venv/bin/activate
python bot_server.py --dev
```

Test parser:
```bash
python3 << 'EOF'
from src.notifications.telegram_bet_handler import parse_bet_command
text = "/bet Pastrnak 50 -110 2026-03-20"
print(parse_bet_command(text))
