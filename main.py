import math
import streamlit as st

# --- Core Functions ---

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
    return (lam ** k * math.exp(-lam)) / math.factorial(k)

def predict_probs(xg_a, xg_b, max_goals=8):
    p_win = p_draw = p_loss = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson_prob(xg_a, i) * poisson_prob(xg_b, j)
            if i > j:    p_win  += p
            elif i == j: p_draw += p
            else:        p_loss += p
    return p_win, p_draw, p_loss

def analyze_team(team_elo, matches):
    weights = [elo_weight(team_elo, m["opp_elo"]) for m in matches]
    xg_att = weighted_average([m["xg_self"] for m in matches], weights)
    xg_def = weighted_average([m["xg_opp"] for m in matches], weights)
    return xg_att, xg_def, weights

# --- UI ---

st.set_page_config(page_title="xG Power Index", layout="wide")
st.title("xG Power Index — Team Comparison")
st.markdown("---")

def team_section(label):
    st.subheader(label)
    name = st.text_input("Team Name", key=f"{label}_name", value=label)
    elo = st.number_input("Elo Rating", min_value=0.0, value=1500.0,
                          step=10.0, key=f"{label}_elo")
    matches = []
    st.markdown("**Last 5 Matches**")
    for i in range(1, 6):
        with st.expander(f"Match {i}"):
            c1, c2, c3 = st.columns(3)
            opp_elo  = c1.number_input("Opponent Elo", min_value=0.0,
                                        value=1500.0, step=10.0,
                                        key=f"{label}_opp_elo_{i}")
            xg_self  = c2.number_input("Your xG", min_value=0.0,
                                        value=1.2, step=0.1,
                                        key=f"{label}_xg_self_{i}")
            xg_opp   = c3.number_input("Opponent xG", min_value=0.0,
                                        value=1.0, step=0.1,
                                        key=f"{label}_xg_opp_{i}")
            matches.append({"opp_elo": opp_elo,
                            "xg_self": xg_self,
                            "xg_opp":  xg_opp})
    return name, elo, matches

col_a, col_b = st.columns(2)
with col_a:
    name_a, elo_a, matches_a = team_section("Team A")
with col_b:
    name_b, elo_b, matches_b = team_section("Team B")

st.markdown("---")

if st.button("Analyze & Predict", use_container_width=True):

    xg_att_a, xg_def_a, weights_a = analyze_team(elo_a, matches_a)
    xg_att_b, xg_def_b, weights_b = analyze_team(elo_b, matches_b)

    # Poisson prediction
    p_win, p_draw, p_loss = predict_probs(xg_att_a, xg_att_b)

    # --- Results ---
    st.markdown("## Results")
    r1, r2 = st.columns(2)

    with r1:
        st.markdown(f"### {name_a}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Attacking xG", f"{xg_att_a:.3f}")
        m2.metric("Defensive xG (conceded)", f"{xg_def_a:.3f}")
        m3.metric("xG Ratio", f"{xg_att_a / (xg_att_a + xg_def_a):.3f}"
                  if (xg_att_a + xg_def_a) > 0 else "N/A")

    with r2:
        st.markdown(f"### {name_b}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Attacking xG", f"{xg_att_b:.3f}")
        m2.metric("Defensive xG (conceded)", f"{xg_def_b:.3f}")
        m3.metric("xG Ratio", f"{xg_att_b / (xg_att_b + xg_def_b):.3f}"
                  if (xg_att_b + xg_def_b) > 0 else "N/A")

    # --- Prediction ---
    st.markdown("---")
    st.markdown("## Match Prediction (Poisson Model)")

    p1, p2, p3 = st.columns(3)
    p1.metric(f"{name_a} Win", f"{p_win*100:.1f}%")
    p2.metric("Draw", f"{p_draw*100:.1f}%")
    p3.metric(f"{name_b} Win", f"{p_loss*100:.1f}%")

    # Verdict
    best = max(p_win, p_draw, p_loss)
    if best == p_win:
        st.success(f"Most likely outcome: **{name_a} wins**")
    elif best == p_draw:
        st.info("Most likely outcome: **Draw**")
    else:
        st.success(f"Most likely outcome: **{name_b} wins**")

    # --- Match Weights ---
    st.markdown("---")
    st.markdown("## Match Weights")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown(f"**{name_a}**")
        st.bar_chart({"Weight": weights_a})
    with w2:
        st.markdown(f"**{name_b}**")
        st.bar_chart({"Weight": weights_b})
