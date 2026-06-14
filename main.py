import math
import streamlit as st
import pandas as pd

# ─── Core Functions ───────────────────────────────────────────────────────────

def elo_weight(team_elo, opponent_elo):
    diff = opponent_elo - team_elo
    return 1 / (1 + math.exp(-diff / 400))

def weighted_average(values, weights):
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total

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
    weights = [elo_weight(team_elo, m["opp_elo"]) for m in matches]
    xg_att = weighted_average([m["xg_self"] for m in matches], weights)
    xg_def = weighted_average([m["xg_opp"]  for m in matches], weights)
    return xg_att, xg_def, weights

def xg_ratio(att, def_):
    total = att + def_
    return f"{att / total:.3f}" if total > 0 else "N/A"

# ─── UI Helpers ───────────────────────────────────────────────────────────────

def team_section(label):
    st.subheader(label)
    name    = st.text_input("Team Name", key=f"{label}_name")
    elo     = st.number_input("Elo Rating", min_value=0, value=1500,
                              step=10, key=f"{label}_elo")
    st.markdown("**Last 5 Matches**")
    matches = []
    for i in range(5):
        c1, c2, c3 = st.columns(3)
        with c1:
            opp_elo  = st.number_input(f"M{i+1} — Opponent Elo",
                                       min_value=0, value=1500,
                                       step=10, key=f"{label}_opp_elo_{i}")
        with c2:
            xg_self  = st.number_input(f"M{i+1} — Your xG",
                                       min_value=0.0, value=1.2,
                                       step=0.1, key=f"{label}_xg_self_{i}")
        with c3:
            xg_opp   = st.number_input(f"M{i+1} — Opponent xG",
                                       min_value=0.0, value=1.0,
                                       step=0.1, key=f"{label}_xg_opp_{i}")
        matches.append({"opp_elo": opp_elo, "xg_self": xg_self, "xg_opp": xg_opp})
    return name, elo, matches

# ─── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="xG Power Index", layout="wide")
st.title("xG Power Index — Team Comparison")

col_a, col_b = st.columns(2)
with col_a:
    name_a, elo_a, matches_a = team_section("Team A")
with col_b:
    name_b, elo_b, matches_b = team_section("Team B")

label_a = name_a or "Team A"
label_b = name_b or "Team B"

if st.button("Analyze & Predict"):

    xg_att_a, xg_def_a, weights_a = analyze_team(elo_a, matches_a)
    xg_att_b, xg_def_b, weights_b = analyze_team(elo_b, matches_b)

    # ── FIX: ترکیب حمله + دفاع حریف ─────────────────────────────────────────
    expected_a = (xg_att_a + xg_def_b) / 2
    expected_b = (xg_att_b + xg_def_a) / 2
    p_win, p_draw, p_loss = predict_probs(expected_a, expected_b)

    # ── Team Stats ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Team Stats")
    s1, s2 = st.columns(2)

    with s1:
        st.markdown(f"**{label_a}**")
        st.metric("Attacking xG",           f"{xg_att_a:.3f}")
        st.metric("Defensive xG (conceded)", f"{xg_def_a:.3f}")
        st.metric("xG Ratio",               xg_ratio(xg_att_a, xg_def_a))

    with s2:
        st.markdown(f"**{label_b}**")
        st.metric("Attacking xG",           f"{xg_att_b:.3f}")
        st.metric("Defensive xG (conceded)", f"{xg_def_b:.3f}")
        st.metric("xG Ratio",               xg_ratio(xg_att_b, xg_def_b))

    # ── Match Prediction ──────────────────────────────────────────────────────
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

    # ── Expected xG ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Expected Goals (Adjusted)")
    e1, e2 = st.columns(2)
    e1.metric(f"{label_a} Expected xG", f"{expected_a:.3f}")
    e2.metric(f"{label_b} Expected xG", f"{expected_b:.3f}")

    # ── Match Weights ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Match Weights")
    w1, w2 = st.columns(2)
    index = [f"M{i+1}" for i in range(5)]

    with w1:
        st.markdown(f"**{label_a}**")
        st.bar_chart(pd.DataFrame({"Weight": weights_a}, index=index))

    with w2:
        st.markdown(f"**{label_b}**")
        st.bar_chart(pd.DataFrame({"Weight": weights_b}, index=index))
