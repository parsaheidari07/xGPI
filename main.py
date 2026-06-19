import math
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def fetch_elo_ratings() -> dict:
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://www.eloratings.net/World.json", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            entries = data if isinstance(data, list) else data.get("teams", [])
            teams = {}
            for entry in entries:
                name = entry.get("name") or entry.get("team", "")
                elo  = int(entry.get("rating") or entry.get("elo", 0))
                if name and elo:
                    teams[name] = elo
            if teams:
                return teams
    except Exception:
        pass
    return {}

def poisson_prob(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lam) * lam**k) / math.factorial(k)

def predict_probs(xg_a, xg_b, max_goals=10):
    p_win = p_draw = p_loss = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson_prob(xg_a, i) * poisson_prob(xg_b, j)
            if i > j:
                p_win += p
            elif i == j:
                p_draw += p
            else:
                p_loss += p
    return p_win, p_draw, p_loss

def analyze_team(team_elo, matches):
    adjusted_atts = []
    adjusted_defs = []
    weights = []
    
    for m in matches:
        diff_factor = math.exp((m["opp_elo"] - team_elo) / 400)
        adj_att = m["xg_self"] * diff_factor
        adj_def = m["xg_opp"] / diff_factor
        
        adjusted_atts.append(adj_att)
        adjusted_defs.append(adj_def)
        weights.append(diff_factor)
        
    xg_att = sum(adjusted_atts) / len(matches) if matches else 0.0
    xg_def = sum(adjusted_defs) / len(matches) if matches else 0.0
    
    return xg_att, xg_def, weights

def xg_ratio(att, def_):
    total = att + def_
    return f"{att / total:.3f}" if total > 0 else "N/A"

def team_section(label, all_teams: dict):
    st.subheader(label)
    
    if all_teams:
        mode = st.radio(
            "Choosing method", ["From list", "Manual"],
            key=f"{label}_mode", horizontal=True,
        )
        use_list = mode == "From list"
    else:
        use_list = False
        st.warning("No live Elo data loaded. Manual input required.")

    if use_list:
        team_names = sorted(all_teams.keys())
        selected   = st.selectbox("Choose team", team_names, key=f"{label}_select")
        name       = selected
        elo        = all_teams[selected]
        st.info(f"Elo Rating: **{elo}**")
    else:
        name = st.text_input("Team Name", key=f"{label}_name")
        elo  = st.number_input("Elo Rating", min_value=0, value=1500,
                               step=10, key=f"{label}_elo")

    st.markdown("**Last 5 Matches**")
    opp_names = sorted(all_teams.keys()) if use_list else []
    matches   = []

    for i in range(5):
        c1, c2, c3 = st.columns(3)
        with c1:
            if use_list:
                opp_sel = st.selectbox(
                    f"M{i+1} — حریف", opp_names,
                    key=f"{label}_opp_sel_{i}",
                )
                opp_elo = all_teams[opp_sel]
                st.caption(f"Elo: {opp_elo}")
            else:
                opp_elo = st.number_input(
                    f"M{i+1} — Opponent Elo",
                    min_value=0, value=1500,
                    step=10, key=f"{label}_opp_elo_{i}",
                )
        with c2:
            xg_self = st.number_input(
                f"M{i+1} — Your xG", min_value=0.0, value=1.2,
                step=0.1, key=f"{label}_xg_self_{i}",
            )
        with c3:
            xg_opp = st.number_input(
                f"M{i+1} — Opponent xG", min_value=0.0, value=1.0,
                step=0.1, key=f"{label}_xg_opp_{i}",
            )
        matches.append({"opp_elo": opp_elo, "xg_self": xg_self, "xg_opp": xg_opp})

    return name, elo, matches

st.set_page_config(page_title="xG Power Index", layout="wide")
st.title("xG Power Index — Team Comparison")

with st.spinner("در حال بارگذاری رتبه‌بندی Elo..."):
    all_teams = fetch_elo_ratings()

if all_teams:
    st.caption(f"Elo data source: **live** — {len(all_teams)} teams loaded")
else:
    st.caption("Elo data source: **failed to load**")

col_a, col_b = st.columns(2)
with col_a:
    name_a, elo_a, matches_a = team_section("Team A", all_teams)
with col_b:
    name_b, elo_b, matches_b = team_section("Team B", all_teams)

label_a = name_a or "Team A"
label_b = name_b or "Team B"

if st.button("Analyze & Predict"):
    xg_att_a, xg_def_a, weights_a = analyze_team(elo_a, matches_a)
    xg_att_b, xg_def_b, weights_b = analyze_team(elo_b, matches_b)

    expected_a = (xg_att_a + xg_def_b) / 2
    expected_b = (xg_att_b + xg_def_a) / 2
    p_win, p_draw, p_loss = predict_probs(expected_a, expected_b)

    st.markdown("---")
    st.subheader("Team Stats (Elo-Adjusted)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**{label_a}**")
        st.metric("Adjusted Attacking xG",            f"{xg_att_a:.3f}")
        st.metric("Adjusted Defensive xG (conceded)", f"{xg_def_a:.3f}")
        st.metric("xG Ratio",                               xg_ratio(xg_att_a, xg_def_a))
    with s2:
        st.markdown(f"**{label_b}**")
        st.metric("Adjusted Attacking xG",            f"{xg_att_b:.3f}")
        st.metric("Adjusted Defensive xG (conceded)", f"{xg_def_b:.3f}")
        st.metric("xG Ratio",                               xg_ratio(xg_att_b, xg_def_b))

    st.markdown("---")
    st.subheader("Match Prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{label_a} Win", f"{p_win*100:.1f}%")
    p2.metric("Draw",           f"{p_draw*100:.1f}%")
    p3.metric(f"{label_b} Win", f"{p_loss*100:.1f}%")

    if p_win > p_loss and p_win > p_draw:
        st.success(f"Prediction: **{label_a}** is favored to win.")
    elif p_loss > p_win and p_loss > p_draw:
        st.success(f"Prediction: **{label_b}** is favored to win.")
    else:
        st.info("Prediction: Match is likely to be a **Draw**.")

    st.markdown("---")
    st.subheader("Expected Goals for Upcoming Match")
    e1, e2 = st.columns(2)
    e1.metric(f"{label_a} Expected goals", f"{expected_a:.3f}")
    e2.metric(f"{label_b} Expected goals", f"{expected_b:.3f}")

    st.markdown("---")
    st.subheader("Match Difficulty & Weight Factors")
    w1, w2 = st.columns(2)
    index  = [f"M{i+1}" for i in range(5)]
    with w1:
        st.markdown(f"**{label_a}** (Higher means tougher opponent)")
        st.bar_chart(pd.DataFrame({"Difficulty Weight": weights_a}, index=index))
    with w2:
        st.markdown(f"**{label_b}** (Higher means tougher opponent)")
        st.bar_chart(pd.DataFrame({"Difficulty Weight": weights_b}, index=index))
