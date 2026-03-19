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


def _build_cheat_sheet_html(pred_df: pd.DataFrame, game_df: pd.DataFrame = None) -> str:
    """Build a quick betting cheat sheet with top 5 picks and best games."""
    top_5 = pred_df.head(5).copy()
    top_5["prob_pct"] = (top_5["goal_probability"] * 100).round(1)
    
    # Build top 5 picks
    picks_html = ""
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        prob = row["prob_pct"]
        if prob >= 40:
            tier_emoji = "🔥"
            tier_color = "#ef4444"
        elif prob >= 38:
            tier_emoji = "🎯"
            tier_color = "#f59e0b"
        else:
            tier_emoji = "👀"
            tier_color = "#3b82f6"
        
        matchup = f"{'vs' if row['is_home'] else '@'} {row['opponent']}"
        gpg = (row["season_goals"] / row["season_gp"]) if row["season_gp"] > 0 else 0
        
        picks_html += f"""        <div class="cheat-pick">
            <div class="pick-rank">{i}</div>
            <div class="pick-info">
                <div class="pick-name">{row['name']} <span class="pick-team">{row['team']}</span></div>
                <div class="pick-matchup">{matchup} • {gpg:.2f} GPG</div>
            </div>
            <div class="pick-prob" style="border-left: 4px solid {tier_color}">
                <div class="pick-emoji">{tier_emoji}</div>
                <div class="pick-percent">{prob:.1f}%</div>
            </div>
        </div>
"""
    
    # Build top games (best game winner picks + best player picks)
    games_html = ""
    
    if game_df is not None and not game_df.empty:
        # Sort by confidence
        top_games = game_df.nlargest(3, "confidence")
        
        for _, g in top_games.iterrows():
            conf = g["confidence"]
            if conf >= 60:
                conf_icon = "🟢"
                conf_label = "STRONG"
            elif conf >= 55:
                conf_icon = "🟡"
                conf_label = "SLIGHT"
            else:
                conf_icon = "⚪"
                conf_label = "LEAN"
            
            winner = g["predicted_winner"]
            matchup = f"{g['away_team']} @ {g['home_team']}"
            probs = f"{g['home_team']} {g['home_win_prob']}% | {g['away_team']} {g['away_win_prob']}%"
            
            games_html += f"""        <div class="cheat-game">
            <div class="game-result">{conf_icon} <strong>{winner} wins</strong></div>
            <div class="game-matchup">{matchup}</div>
            <div class="game-probs">{probs}</div>
        </div>
"""
    
    return f"""    <section class="cheat-sheet">
        <div class="cheat-header">
            <h2>📋 BETTING CHEAT SHEET</h2>
            <p>Top 5 picks + Best games. Copy these to your sportsbook.</p>
        </div>
        
        <div class="cheat-grid">
            <div class="cheat-column">
                <h3>🎯 TOP 5 PLAYER PICKS</h3>
{picks_html}
            </div>
            
            <div class="cheat-column">
                <h3>🏆 TOP GAMES (By Confidence)</h3>
{games_html if games_html else '                <p style="color: #94a3b8; padding: 1rem;">No game data available</p>'}
            </div>
        </div>
    </section>
"""


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

    return f"""        <h2>🏆 All Game Winner Predictions</h2>
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
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
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
        
        /* CHEAT SHEET STYLES */
        .cheat-sheet {{
            margin: 2.5rem 0;
            background: linear-gradient(135deg, #1a1f35 0%, #0d1322 100%);
            border: 2px solid #22c55e;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(34, 197, 94, 0.15);
        }}
        .cheat-header {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .cheat-header h2 {{
            font-size: 1.8rem;
            color: #22c55e;
            margin-bottom: 0.5rem;
        }}
        .cheat-header p {{
            color: #94a3b8;
            font-size: 0.95rem;
        }}
        .cheat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}
        @media (max-width: 900px) {{
            .cheat-grid {{ grid-template-columns: 1fr; }}
        }}
        
        .cheat-column h3 {{
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #f1f5f9;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #334155;
        }}
        
        .cheat-pick {{
            display: flex;
            align-items: center;
            gap: 1rem;
            background: #1a1f35;
            border: 1px solid #2a3352;
            border-radius: 10px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            transition: all 0.2s;
        }}
        .cheat-pick:hover {{
            background: #1e293b;
            border-color: #475569;
        }}
        .pick-rank {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #22c55e;
            min-width: 30px;
            text-align: center;
        }}
        .pick-info {{
            flex: 1;
        }}
        .pick-name {{
            font-weight: 700;
            color: #f1f5f9;
            font-size: 0.95rem;
        }}
        .pick-team {{
            color: #94a3b8;
            font-weight: 600;
            margin-left: 0.5rem;
        }}
        .pick-matchup {{
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}
        .pick-prob {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            background: #0f172a;
            border-radius: 8px;
            min-width: 80px;
        }}
        .pick-emoji {{
            font-size: 1.3rem;
        }}
        .pick-percent {{
            font-weight: 800;
            font-size: 1.1rem;
            color: #22c55e;
        }}
        
        .cheat-game {{
            background: #1a1f35;
            border: 1px solid #2a3352;
            border-radius: 10px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            transition: all 0.2s;
        }}
        .cheat-game:hover {{
            background: #1e293b;
            border-color: #475569;
        }}
        .game-result {{
            font-weight: 700;
            color: #f1f5f9;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }}
        .game-matchup {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 0.3rem;
        }}
        .game-probs {{
            font-size: 0.8rem;
            color: #64748b;
        }}
        
        /* TABLE STYLES */
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

{_build_cheat_sheet_html(pred_df, game_df)}

        <h2>🎯 Full Player Rankings (Top {top_n})</h2>
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
