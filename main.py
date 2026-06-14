import math
import streamlit as st

def elo_weight(team_elo, opponent_elo):
    diff = opponent_elo - team_elo
    return 1 / (1 + math.exp(-diff / 400))

def weighted_average(values, weights):
    total_w = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total_w

# ─── UI ───────────────────────────────────────────────
st.title("Team Strength Analysis with xG")
st.markdown("---")

team_elo = st.number_input("Team Elo Rating", min_value=0.0, value=1500.0, step=10.0)

st.markdown("### Last 5 Matches")

matches = []
for i in range(1, 6):
    with st.expander(f"Match {i}", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            opp_elo = st.number_input("Opponent Elo", min_value=0.0, value=1500.0,
                                      step=10.0, key=f"elo_{i}")
        with col2:
            xg_self = st.number_input("Your xG", min_value=0.0, value=1.0,
                                      step=0.1, key=f"xgs_{i}")
        with col3:
            xg_opp = st.number_input("Opponent xG", min_value=0.0, value=1.0,
                                     step=0.1, key=f"xgo_{i}")
        matches.append({"opp_elo": opp_elo, "xg_self": xg_self, "xg_opp": xg_opp})

st.markdown("---")

if st.button("Calculate Team Strength", use_container_width=True):
    weights = [elo_weight(team_elo, m["opp_elo"]) for m in matches]
    xg_att  = weighted_average([m["xg_self"] for m in matches], weights)
    xg_def  = weighted_average([m["xg_opp"]  for m in matches], weights)
    power   = xg_att * xg_def

    st.markdown("## Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Attacking xG", f"{xg_att:.3f}")
    c2.metric("Opponent xG (Defensive)", f"{xg_def:.3f}")
    c3.metric("Team Strength", f"{power:.3f}")

    st.markdown("### Match Weights")
    weight_data = {
        "Match":         [f"Match {i+1}" for i in range(5)],
        "Opponent Elo":  [m["opp_elo"]  for m in matches],
        "Your xG":       [m["xg_self"]  for m in matches],
        "Opponent xG":   [m["xg_opp"]   for m in matches],
        "Weight":        [round(w, 4)   for w in weights],
    }
    st.dataframe(weight_data, use_container_width=True)

    st.bar_chart({"Match Weights": weights})