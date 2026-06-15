import math
import streamlit as st
import pandas as pd

# ─── Static Elo Data (fallback) ───────────────────────────────────────────────

STATIC_ELO = {
    "Spain": 2157, "Argentina": 2115, "France": 2063, "England": 2024,
    "Portugal": 1989, "Colombia": 1982, "Brazil": 1978, "Netherlands": 1944,
    "Germany": 1939, "Norway": 1914, "Croatia": 1912, "Japan": 1910,
    "Belgium": 1894, "Uruguay": 1892, "Ecuador": 1890, "Mexico": 1881,
    "Denmark": 1869, "Italy": 1869, "Switzerland": 1865, "Senegal": 1860,
    "Turkey": 1849, "Morocco": 1840, "Australia": 1839, "Austria": 1830,
    "Scotland": 1794, "South Korea": 1786, "Paraguay": 1780, "Ukraine": 1780,
    "United States": 1780, "Algeria": 1772, "Iran": 1772, "Canada": 1767,
    "Nigeria": 1767, "Sweden": 1755, "Greece": 1744, "Ivory Coast": 1743,
    "Serbia": 1734, "Venezuela": 1733, "Panama": 1730, "Chile": 1717,
    "Kosovo": 1715, "Uzbekistan": 1714, "Czechia": 1712, "Hungary": 1710,
    "Poland": 1710, "Peru": 1700, "Ireland": 1699, "Egypt": 1696,
    "Wales": 1682, "Slovenia": 1682, "Jordan": 1680, "Slovakia": 1667,
    "Georgia": 1654, "DR Congo": 1652, "Israel": 1647, "Romania": 1639,
    "Bolivia": 1621, "Bosnia and Herzegovina": 1616, "Albania": 1616,
    "Cameroon": 1614, "Costa Rica": 1608, "Iraq": 1607,
    "Northern Ireland": 1605, "North Macedonia": 1589, "Mali": 1588,
    "Tunisia": 1585, "Cape Verde": 1578, "Saudi Arabia": 1576,
    "Honduras": 1570, "Iceland": 1568, "New Zealand": 1562, "Angola": 1542,
    "United Arab Emirates": 1540, "Finland": 1536, "Haiti": 1536,
    "Burkina Faso": 1529, "Jamaica": 1527, "Belarus": 1522,
    "South Africa": 1511, "Ghana": 1510, "Guatemala": 1504, "Oman": 1480,
    "Syria": 1479, "Palestine": 1465, "Guinea": 1463, "Montenegro": 1461,
    "Bulgaria": 1458, "Luxembourg": 1450, "Qatar": 1447, "Suriname": 1431,
    "Kazakhstan": 1428, "Curacao": 1427, "China": 1424, "Libya": 1420,
    "Gambia": 1419, "Bahrain": 1414, "Benin": 1405, "Gabon": 1401,
    "Uganda": 1394, "Trinidad and Tobago": 1386, "Faroe Islands": 1386,
    "Niger": 1382, "Madagascar": 1380, "Togo": 1379, "Thailand": 1376,
    "North Korea": 1375, "Comoros": 1374, "Armenia": 1373, "Zimbabwe": 1372,
    "Indonesia": 1372, "Zambia": 1371, "Kenya": 1363, "Estonia": 1360,
    "Vietnam": 1351, "Sudan": 1350, "El Salvador": 1342, "Mozambique": 1342,
    "Sierra Leone": 1341, "Rwanda": 1336, "Nicaragua": 1333, "Kuwait": 1332,
    "Mauritania": 1329, "Azerbaijan": 1322, "Cyprus": 1314, "Tanzania": 1313,
    "Liberia": 1304, "Namibia": 1303, "Kyrgyzstan": 1295, "Malaysia": 1293,
    "Guyana": 1292, "Lebanon": 1288, "Latvia": 1288, "Ethiopia": 1287,
    "Tajikistan": 1285, "Burundi": 1285, "Dominican Republic": 1283,
    "Lithuania": 1279, "Moldova": 1270, "Botswana": 1267, "Malta": 1255,
    "Guinea-Bissau": 1248, "Cuba": 1239, "Malawi": 1239,
    "Central African Republic": 1236, "Turkmenistan": 1209, "Congo": 1207,
    "Eritrea": 1201, "Lesotho": 1198, "Yemen": 1195, "Philippines": 1179,
    "Eswatini": 1148, "Papua New Guinea": 1135, "Singapore": 1134,
    "India": 1128, "Vanuatu": 1118, "Bermuda": 1117, "South Sudan": 1109,
    "Fiji": 1104, "Hong Kong": 1101, "Grenada": 1098,
}

# ─── Elo Fetcher ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_elo_ratings() -> dict:
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://www.eloratings.net/World.json",
                         headers=headers, timeout=10)
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
    # fallback به داده استاتیک
    return STATIC_ELO

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
    xg_att  = weighted_average([m["xg_self"] for m in matches], weights)
    xg_def  = weighted_average([m["xg_opp"]  for m in matches], weights)
    return xg_att, xg_def, weights

def xg_ratio(att, def_):
    total = att + def_
    return f"{att / total:.3f}" if total > 0 else "N/A"

# ─── UI Helpers ───────────────────────────────────────────────────────────────

def team_section(label, all_teams: dict):
    st.subheader(label)
    mode = st.radio(
        "روش انتخاب تیم", ["از لیست", "دستی"],
        key=f"{label}_mode", horizontal=True,
    )
    use_list = mode == "از لیست"

    if use_list:
        team_names = sorted(all_teams.keys())
        selected   = st.selectbox("انتخاب تیم", team_names, key=f"{label}_select")
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

# ─── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="xG Power Index", layout="wide")
st.title("xG Power Index — Team Comparison")

with st.spinner("در حال بارگذاری رتبه‌بندی Elo..."):
    all_teams = fetch_elo_ratings()

source = "live" if all_teams is not STATIC_ELO else "static"
st.caption(f"Elo data source: **{source}** — {len(all_teams)} teams loaded")

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
    st.subheader("Team Stats")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**{label_a}**")
        st.metric("Attacking xG",            f"{xg_att_a:.3f}")
        st.metric("Defensive xG (conceded)", f"{xg_def_a:.3f}")
        st.metric("xG Ratio",                xg_ratio(xg_att_a, xg_def_a))
    with s2:
        st.markdown(f"**{label_b}**")
        st.metric("Attacking xG",            f"{xg_att_b:.3f}")
        st.metric("Defensive xG (conceded)", f"{xg_def_b:.3f}")
        st.metric("xG Ratio",                xg_ratio(xg_att_b, xg_def_b))

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
    st.subheader("Expected Goals (Adjusted)")
    e1, e2 = st.columns(2)
    e1.metric(f"{label_a} Expected xG", f"{expected_a:.3f}")
    e2.metric(f"{label_b} Expected xG", f"{expected_b:.3f}")

    st.markdown("---")
    st.subheader("Match Weights")
    w1, w2 = st.columns(2)
    index  = [f"M{i+1}" for i in range(5)]
    with w1:
        st.markdown(f"**{label_a}**")
        st.bar_chart(pd.DataFrame({"Weight": weights_a}, index=index))
    with w2:
        st.markdown(f"**{label_b}**")
        st.bar_chart(pd.DataFrame({"Weight": weights_b}, index=index))
