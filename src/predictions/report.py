"""
HTML report generator for daily predictions.

Creates a shareable, styled HTML page with tonight's picks.
Open it in any browser, screenshot it, or send the file to friends.
"""

import os
from datetime import datetime

import pandas as pd


REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _tier_label(prob: float) -> str:
    """Assign a tier based on goal probability."""
    if prob >= 0.72:
        return "🔥 FIRE"
    if prob >= 0.68:
        return "🎯 STRONG"
    if prob >= 0.64:
        return "👀 WATCH"
    return "📋 LONG SHOT"


def _tier_class(prob: float) -> str:
    if prob >= 0.72:
        return "fire"
    if prob >= 0.68:
        return "strong"
    if prob >= 0.64:
        return "watch"
    return "longshot"


def _build_game_winner_html(game_df: pd.DataFrame) -> str:
    """Build HTML section for game winner predictions."""
    if game_df is None or game_df.empty:
        return ""

    rows_html = ""
    for _, g in game_df.iterrows():
        is_home_fav = g["home_win_prob"] > 50
        winner = g["predicted_winner"]
        conf = g["confidence"]

        if conf >= 60:
            conf_class = "fire"
            conf_icon = "🟢"
        elif conf >= 55:
            conf_class = "strong"
            conf_icon = "🟡"
        else:
            conf_class = "longshot"
            conf_icon = "⚪"

        venue_icon = "🏠" if is_home_fav else "✈️"
        bar_width = g["home_win_prob"]

        home_pp = g.get('home_pp_pct', 0.20)
        away_pp = g.get('away_pp_pct', 0.20)
        pp_display = (
            f"PP: {g['home_team']} {home_pp*100 if home_pp < 1 else home_pp:.1f}% | "
            f"{g['away_team']} {away_pp*100 if away_pp < 1 else away_pp:.1f}%"
        )

        rows_html += f"""        <div class="game-winner-card {conf_class}">
            <div class="gw-matchup">{g['away_team']} @ {g['home_team']}</div>
            <div class="gw-pick">{conf_icon} {venue_icon} <strong>{winner}</strong> ({conf}%)</div>
            <div class="gw-bar-container">
                <div class="gw-bar-home" style="width: {bar_width}%">{g['home_team']} {g['home_win_prob']}%</div>
                <div class="gw-bar-away" style="width: {100 - bar_width}%">{g['away_team']} {g['away_win_prob']}%</div>
            </div>
            <div class="gw-special-teams">{pp_display}</div>
        </div>
"""

    return f"""        <h2>🏆 Game Winner Predictions</h2>
        <div class="gw-grid">
{rows_html}
        </div>"""


def generate_html_report(pred_df: pd.DataFrame, top_n: int = 30, game_df: pd.DataFrame = None) -> str:
    """
    Generate a full HTML report from prediction data.

    Args:
        pred_df: DataFrame from predict_tonight()
        top_n: Number of players to include
        game_df: Optional DataFrame from predict_game_winners()

    Returns:
        Path to the generated HTML file.
    """
    _ensure_report_dir()

    today = datetime.now().strftime("%Y-%m-%d")
    display = pred_df.head(top_n).copy()
    display["prob_pct"] = (display["goal_probability"] * 100).round(1)
    display["gpg"] = (display["season_goals"] / display["season_gp"].clip(lower=1)).round(2)
    display["matchup"] = display.apply(
        lambda r: f"{'vs' if r['is_home'] else '@'} {r['opponent']}", axis=1
    )
    display["tier"] = display["goal_probability"].apply(_tier_label)
    display["tier_class"] = display["goal_probability"].apply(_tier_class)

    # Streak indicator
    def _streak_badge(row):
        parts = []
        if row.get("sell_high", 0):
            pdo = row.get("pdo", 0)
            parts.append(f'<span class="streak-sell">📉 PDO {pdo:.0f}</span>')
        if row.get("is_hot", 0):
            parts.append(f'<span class="streak-hot">🔥 {int(row.get("goal_streak", 0))}G</span>')
        if row.get("drought", 0) >= 5:
            parts.append(f'<span class="streak-cold">❄️ {int(row.get("drought", 0))}G</span>')
        return " ".join(parts)

    display["streak_badge"] = display.apply(_streak_badge, axis=1)
    display["goalie_info"] = display.get("opp_goalie_name", pd.Series([""] * len(display)))
    display["b2b_flag"] = display.apply(
        lambda r: "⚠️ B2B" if r.get("is_back_to_back", 0) else "", axis=1
    )

    # Build player rows
    player_rows = ""
    for i, (_, row) in enumerate(display.iterrows(), 1):
        player_rows += f"""        <tr class="{row['tier_class']}">
            <td class="rank">{i}</td>
            <td class="player">{row['name']} {row['streak_badge']} {row['b2b_flag']}</td>
            <td>{row['team']}</td>
            <td>{row['position']}</td>
            <td>{row['matchup']}</td>
            <td class="prob"><div class="prob-bar" style="width: {row['prob_pct']}%">{row['prob_pct']}%</div></td>
            <td>{row['gpg']}</td>
            <td>{row['rolling_goals_avg']:.2f}</td>
            <td>{row['rolling_shots_avg']:.1f}</td>
            <td>{int(row['season_goals'])}</td>
            <td class="tier-badge">{row['tier']}</td>
            <td class="goalie-col">{row.get('goalie_info', '')}</td>
        </tr>
"""

    # Build per-game breakdown
    game_cards = ""
    seen_matchups = set()
    for _, row in pred_df.iterrows():
        if row["is_home"]:
            matchup_key = f"{row['team']}_vs_{row['opponent']}"
            matchup_display = f"{row['opponent']} @ {row['team']}"
        else:
            matchup_key = f"{row['opponent']}_vs_{row['team']}"
            matchup_display = f"{row['team']} @ {row['opponent']}"

        if matchup_key in seen_matchups:
            continue
        seen_matchups.add(matchup_key)

        # Get top 3 from each team in this matchup
        home_team = row["team"] if row["is_home"] else row["opponent"]
        away_team = row["opponent"] if row["is_home"] else row["team"]

        home_players = pred_df[pred_df["team"] == home_team].head(3)
        away_players = pred_df[pred_df["team"] == away_team].head(3)

        home_list = "".join(
            f"<li>{r['name']} <span class='pct'>{r['goal_probability']*100:.0f}%</span></li>"
            for _, r in home_players.iterrows()
        )
        away_list = "".join(
            f"<li>{r['name']} <span class='pct'>{r['goal_probability']*100:.0f}%</span></li>"
            for _, r in away_players.iterrows()
        )

        game_cards += f"""        <div class="game-card">
            <div class="game-header">{matchup_display}</div>
            <div class="game-teams">
                <div class="team-col">
                    <h4>🏠 {home_team}</h4>
                    <ol>{home_list}</ol>
                </div>
                <div class="team-col">
                    <h4>✈️ {away_team}</h4>
                    <ol>{away_list}</ol>
                </div>
            </div>
        </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏒 Snipe Tracker — {today}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e17;
            color: #e2e8f0;
            padding: 2rem;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, #1a1f35 0%, #0d1322 100%);
            border-radius: 16px;
            border: 1px solid #2a3352;
        }}
        header h1 {{ font-size: 2.2rem; margin-bottom: 0.5rem; }}
        header .date {{ color: #94a3b8; font-size: 1.1rem; }}
        header .subtitle {{ color: #64748b; font-size: 0.9rem; margin-top: 0.5rem; }}
        .disclaimer {{
            background: #1c1917;
            border: 1px solid #78350f;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 1.5rem 0;
            font-size: 0.85rem;
            color: #fbbf24;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.9rem;
        }}
        th {{
            background: #1e293b;
            color: #94a3b8;
            padding: 0.75rem 0.5rem;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            border-bottom: 2px solid #334155;
        }}
        td {{
            padding: 0.6rem 0.5rem;
            border-bottom: 1px solid #1e293b;
        }}
        tr:hover {{ background: #1e293b; }}
        tr.fire td {{ border-left: 3px solid #ef4444; }}
        tr.strong td {{ border-left: 3px solid #f59e0b; }}
        tr.watch td {{ border-left: 3px solid #3b82f6; }}
        tr.longshot td {{ border-left: 3px solid #475569; }}
        .rank {{ color: #64748b; font-weight: 600; width: 30px; }}
        .player {{ font-weight: 600; color: #f1f5f9; }}
        .prob {{ width: 120px; }}
        .prob-bar {{
            background: linear-gradient(90deg, #22c55e, #16a34a);
            color: #fff;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.85rem;
            text-align: right;
            min-width: 50px;
            display: inline-block;
        }}
        tr.fire .prob-bar {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        tr.strong .prob-bar {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
        tr.watch .prob-bar {{ background: linear-gradient(90deg, #3b82f6, #2563eb); }}
        .tier-badge {{ font-size: 0.8rem; white-space: nowrap; }}
        h2 {{
            font-size: 1.4rem;
            margin: 2.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #1e293b;
        }}
        .games-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .game-card {{
            background: #1a1f35;
            border: 1px solid #2a3352;
            border-radius: 12px;
            overflow: hidden;
        }}
        .game-header {{
            background: #1e293b;
            padding: 0.75rem 1rem;
            font-weight: 700;
            font-size: 1.05rem;
            text-align: center;
            border-bottom: 1px solid #2a3352;
        }}
        .game-teams {{ display: flex; }}
        .team-col {{
            flex: 1;
            padding: 0.75rem 1rem;
        }}
        .team-col:first-child {{ border-right: 1px solid #2a3352; }}
        .team-col h4 {{ margin-bottom: 0.5rem; font-size: 0.95rem; }}
        .team-col ol {{ padding-left: 1.2rem; }}
        .team-col li {{ margin-bottom: 0.3rem; font-size: 0.9rem; }}
        .pct {{ color: #22c55e; font-weight: 700; }}
        .streak-hot {{ background: #7f1d1d; color: #fca5a5; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .streak-cold {{ background: #1e3a5f; color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .goalie-col {{ font-size: 0.85rem; color: #94a3b8; }}
        .gw-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .game-winner-card {{
            background: #1a1f35;
            border: 1px solid #2a3352;
            border-radius: 12px;
            padding: 1rem;
        }}
        .game-winner-card.fire {{ border-left: 3px solid #22c55e; }}
        .game-winner-card.strong {{ border-left: 3px solid #f59e0b; }}
        .game-winner-card.longshot {{ border-left: 3px solid #475569; }}
        .gw-matchup {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 0.4rem; }}
        .gw-pick {{ font-size: 0.95rem; margin-bottom: 0.6rem; color: #94a3b8; }}
        .gw-pick strong {{ color: #f1f5f9; }}
        .gw-bar-container {{ display: flex; border-radius: 6px; overflow: hidden; height: 24px; font-size: 0.75rem; }}
        .gw-bar-home {{ background: #3b82f6; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 600; }}
        .gw-bar-away {{ background: #64748b; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 600; }}
        .gw-special-teams {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem; text-align: center; }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            padding: 1.5rem;
            color: #475569;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏒 Snipe Tracker</h1>
            <div class="date">Goal Scorer Predictions — {today}</div>
            <div class="subtitle">Model: Logistic Regression · AUC: 0.711 · {len(pred_df)} players analyzed</div>
        </header>

        <div class="disclaimer">
            ⚠️ For entertainment purposes. Model probabilities are relative rankings,
            not absolute odds. Never bet more than you can afford to lose.
        </div>

        <h2>🎯 Top {top_n} Most Likely Goal Scorers</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Pos</th>
                    <th>Matchup</th>
                    <th>Goal Prob</th>
                    <th>GPG</th>
                    <th>Roll G/Gm</th>
                    <th>Roll S/Gm</th>
                    <th>Season G</th>
                    <th>Tier</th>
                    <th>vs Goalie</th>
                </tr>
            </thead>
            <tbody>
{player_rows}
            </tbody>
        </table>

        <h2>📋 Breakdown by Game</h2>
        <div class="games-grid">
{game_cards}
        </div>

{_build_game_winner_html(game_df) if game_df is not None and not game_df.empty else ''}

        <footer>
            Snipe Tracker · Built with Python, scikit-learn & the NHL API<br>
            Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </footer>
    </div>
</body>
</html>"""

    filepath = os.path.join(REPORT_DIR, f"picks_{today}.html")
    with open(filepath, "w") as f:
        f.write(html)

    print(f"\n📄 Report saved to: {filepath}")
    return filepath
