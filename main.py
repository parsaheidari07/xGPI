import math
import streamlit as st
import pandas as pd

UPDATED_ELO = {
    "Spain": 2129, "Argentina": 2128, "France": 2084, "England": 2055,
    "Colombia": 1998, "Brazil": 1978, "Portugal": 1967, "Netherlands": 1944,
    "Germany": 1939, "Norway": 1929, "Japan": 1910, "Mexico": 1896,
    "Ecuador": 1890, "Switzerland": 1885, "Croatia": 1881, "Belgium": 1879,
    "Uruguay": 1870, "Denmark": 1869, "Italy": 1869, "Austria": 1857,
    "Turkey": 1849, "Morocco": 1840, "Australia": 1839, "Senegal": 1839,
    "Scotland": 1794, "Paraguay": 1780, "Ukraine": 1780, "United States": 1780,
    "Canada": 1777, "Russia": 1772, "South Korea": 1771, "Nigeria": 1767,
    "Algeria": 1759, "Iran": 1756, "Sweden": 1755, "Greece": 1744,
    "Ivory Coast": 1743, "Serbia": 1734, "Venezuela": 1733, "Chile": 1717,
    "Kosovo": 1715, "Egypt": 1711, "Hungary": 1710, "Poland": 1710,
    "Peru": 1699, "Ireland": 1699, "Uzbekistan": 1698, "Czechia": 1696,
    "Panama": 1683, "Wales": 1682, "Slovenia": 1682, "DR Congo": 1674,
    "Slovakia": 1667, "Georgia": 1654, "Jordan": 1653, "Israel": 1647,
    "Romania": 1639, "Bolivia": 1621, "Albania": 1616, "Cameroon": 1614,
    "Costa Rica": 1608, "Northern Ireland": 1606, "Cape Verde": 1606, "Saudi Arabia": 1598,
    "Bosnia and Herzegovina": 1596, "Iraq": 1592, "North Macedonia": 1589, "Mali": 1588,
    "Tunisia": 1585, "New Zealand": 1578, "Honduras": 1570, "Iceland": 1568,
    "Ghana": 1557, "Angola": 1542, "United Arab Emirates": 1540, "Finland": 1536,
    "Haiti": 1536, "Burkina Faso": 1529, "South Africa": 1527, "Jamaica": 1527,
    "Belarus": 1522, "Guatemala": 1504, "Syria": 1479, "Oman": 1479,
    "Palestine": 1465, "Guinea": 1463, "Montenegro": 1461, "Bulgaria": 1458,
    "Luxembourg": 1450, "Northern Cyprus": 1442, "Qatar": 1437, "Suriname": 1431,
    "Kazakhstan": 1428, "Curaçao": 1427, "China": 1424, "Kurdistan": 1424,
    "Libya": 1420, "Gambia": 1419, "Bahrain": 1414, "Benin": 1405,
    "Gabon": 1401, "Uganda": 1394, "Trinidad and Tobago": 1386,
}


@st.cache_data(ttl=3600)
def fetch_elo_ratings():
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://www.eloratings.net/World.json",
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            entries = data if isinstance(data, list) else data.get("teams", [])
            teams = {}
            for entry in entries:
                name = entry.get("name") or entry.get("team", "")
                elo = int(entry.get("rating") or entry.get("elo", 0))
                if name and elo:
                    teams[name] = elo
            if teams:
                return teams, "live"
    except Exception:
        pass
    return UPDATED_ELO, "fallback"


def poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lam) * lam**k) / math.factorial(k)


def predict_probs(xg_a: float, xg_b: float, max_goals: int = 10):
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
                
    # نرمال‌سازی احتمالات برای اینکه مجموع دقیقاً ۱۰۰٪ شود
    total_p = p_win + p_draw + p_loss
    if total_p > 0:
        p_win /= total_p
        p_draw /= total_p
        p_loss /= total_p
        
    return p_win, p_draw, p_loss


def analyze_team(team_elo: int, matches: list):
    """
    Weighted average of raw xG values using both Elo difficulty AND Recency (Time Decay).
    - Elo weight = exp((opp_elo - team_elo) / 400)
    - Time weight = 1.0 down to 0.6 (Latest match gets 100%, 5th match gets 60% weight)
    - Final Weight = Elo weight * Time weight
    """
    if not matches:
        return 0.0, 0.0, []

    weights = []
    for i, m in enumerate(matches):
        # ۱. وزن سختی حریف بر اساس رتبه الیور
        diff_factor = math.exp((m["opp_elo"] - team_elo) / 400)
        
        # ۲. وزن زمانی (کاهش خطی تاثیر بازی‌های قدیمی‌تر)
        # M1 (i=0) -> 1.0 | M2 (i=1) -> 0.9 | ... | M5 (i=4) -> 0.6
        time_factor = 1.0 - (i * 0.1)
        
        # ترکیب هر دو فاکتور به عنوان وزن نهایی بازی
        final_weight = diff_factor * time_factor
        weights.append(final_weight)

    total_w = sum(weights) if sum(weights) > 0 else 1.0

    # میانگین وزنی متغیرهای xG خودی و حریف
    xg_att = sum(m["xg_self"] * w for m, w in zip(matches, weights)) / total_w
    xg_def = sum(m["xg_opp"]  * w for m, w in zip(matches, weights)) / total_w

    return xg_att, xg_def, weights


def combine_xg(xg_att: float, xg_def_opp: float) -> float:
    """
    Geometric mean of a team's attacking xG and the opponent's defensive xG.
    Falls back to arithmetic mean if either value is non-positive.
    """
    if xg_att > 0 and xg_def_opp > 0:
        return math.sqrt(xg_att * xg_def_opp)
    return (xg_att + xg_def_opp) / 2


def xg_dominance(att: float, def_: float) -> str:
    """Attack / Defense ratio. > 1 means team creates more than it concedes."""
    return f"{att / def_:.3f}" if def_ > 0 else "N/A"


def team_section(label: str, all_teams: dict):
    st.subheader(label)

    mode = st.radio(
        "Choosing method",
        ["From list", "Manual"],
        key=f"{label}_mode",
        horizontal=True,
    )
    use_list = mode == "From list"

    if use_list:
        team_names = sorted(all_teams.keys())
        selected = st.selectbox("Choose team", team_names, key=f"{label}_select")
        name = selected
        elo = all_teams[selected]
        st.info(f"Elo Rating: **{elo}**")
    else:
        name = st.text_input("Team Name", key=f"{label}_name")
        elo = st.number_input(
            "Elo Rating", min_value=0, value=1500, step=10, key=f"{label}_elo"
        )

    st.markdown("**Last 5 Matches (M1 = Most Recent, M5 = Oldest)**")
    opp_names = sorted(all_teams.keys())
    matches = []

    for i in range(5):
        c1, c2, c3 = st.columns(3)
        with c1:
            if use_list:
                opp_sel = st.selectbox(
                    f"M{i+1} — Opponent",
                    opp_names,
                    key=f"{label}_opp_sel_{i}",
                )
                opp_elo = all_teams[opp_sel]
                st.caption(f"Elo: {opp_elo}")
            else:
                opp_elo = st.number_input(
                    f"M{i+1} — Opponent Elo",
                    min_value=0,
                    value=1500,
                    step=10,
                    key=f"{label}_opp_elo_{i}",
                )
        with c2:
            xg_self = st.number_input(
                f"M{i+1} — Your xG",
                min_value=0.0,
                value=1.2,
                step=0.1,
                key=f"{label}_xg_self_{i}",
            )
        with c3:
            xg_opp = st.number_input(
                f"M{i+1} — Opponent xG",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"{label}_xg_opp_{i}",
            )
        matches.append({"opp_elo": opp_elo, "xg_self": xg_self, "xg_opp": xg_opp})

    return name, elo, matches


# ── App layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="xG Power Index", layout="wide")
st.title("xG Power Index — Team Comparison")

with st.spinner("Loading Elo ratings..."):
    all_teams, source = fetch_elo_ratings()

if source == "live":
    st.caption(f"Elo data source: **Live API** — {len(all_teams)} teams loaded")
else:
    st.caption("Elo data source: **Fallback Mode** (connection failed)")

col_a, col_b = st.columns(2)
with col_a:
    name_a, elo_a, matches_a = team_section("Team A", all_teams)
with col_b:
    name_b, elo_b, matches_b = team_section("Team B", all_teams)

label_a = name_a or "Team A"
label_b = name_b or "Team B"

if st.button("Analyze & Predict", type="primary"):
    xg_att_a, xg_def_a, weights_a = analyze_team(elo_a, matches_a)
    xg_att_b, xg_def_b, weights_b = analyze_team(elo_b, matches_b)

    # Expected goals for the upcoming match:
    expected_a = combine_xg(xg_att_a, xg_def_b)
    expected_b = combine_xg(xg_att_b, xg_def_a)

    p_win, p_draw, p_loss = predict_probs(expected_a, expected_b)

    # ── Team stats ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Team Stats (Elo & Recency Adjusted)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**{label_a}**")
        st.metric("Attacking xG (weighted avg)",      f"{xg_att_a:.3f}")
        st.metric("Defensive xG conceded (weighted)", f"{xg_def_a:.3f}")
        st.metric("xG Dominance (att / def)",         xg_dominance(xg_att_a, xg_def_a))
    with s2:
        st.markdown(f"**{label_b}**")
        st.metric("Attacking xG (weighted avg)",      f"{xg_att_b:.3f}")
        st.metric("Defensive xG conceded (weighted)", f"{xg_def_b:.3f}")
        st.metric("xG Dominance (att / def)",         xg_dominance(xg_att_b, xg_def_b))

    # ── Match prediction ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Match Prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"{label_a} Win", f"{p_win  * 100:.1f}%")
    p2.metric("Draw",           f"{p_draw * 100:.1f}%")
    p3.metric(f"{label_b} Win", f"{p_loss * 100:.1f}%")

    if p_win > p_loss and p_win > p_draw:
        st.success(f"Prediction: **{label_a}** is favored to win.")
    elif p_loss > p_win and p_loss > p_draw:
        st.success(f"Prediction: **{label_b}** is favored to win.")
    else:
        st.info("Prediction: Match is likely to be a **Draw**.")

    # ── Expected goals ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Expected Goals for Upcoming Match")
    e1, e2 = st.columns(2)
    e1.metric(f"{label_a} Expected Goals", f"{expected_a:.3f}")
    e2.metric(f"{label_b} Expected Goals", f"{expected_b:.3f}")

    # ── Total Combined Weights ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Total Match Weights (Elo × Recency)")
    st.caption(
        "This chart shows the combined weight for each match. "
        "M1 is heavily boosted by being recent. Higher bars mean the match is highly influential due to either recency, opponent difficulty, or both."
    )
    w1, w2 = st.columns(2)
    index = [f"M{i+1}" for i in range(5)]
    with w1:
        st.markdown(f"**{label_a}** — Combined Weights")
        st.bar_chart(pd.DataFrame({"Total Weight": weights_a}, index=index))
    with w2:
        st.markdown(f"**{label_b}** — Combined Weights")
        st.bar_chart(pd.DataFrame({"Total Weight": weights_b}, index=index))
