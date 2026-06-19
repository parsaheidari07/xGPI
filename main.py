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
    return p_win, p_draw, p_loss


def analyze_team(team_elo: int, matches: list, days_decay: float = 0.05):
    """
    میانگین وزن‌دار xG با در نظر گرفتن قدرت حریف و زمان بازی.
    
    وزن = exp((opp_elo - team_elo) / 400) × exp(-days_decay × days_ago)
    
    حریف قوی‌تر و بازی‌های اخیر وزن بیشتری دارند.
    """
    if not matches:
        return 0.0, 0.0, []

    weights = []
    for m in matches:
        diff_factor = math.exp((m["opp_elo"] - team_elo) / 400)
        time_factor = math.exp(-days_decay * m.get("days_ago", 0))
        weights.append(diff_factor * time_factor)

    total_w = sum(weights)
    if total_w == 0:
        return 0.0, 0.0, weights

    # نرمال‌سازی وزن‌ها
    normalized_weights = [w / total_w for w in weights]

    # میانگین وزن‌دار
    xg_att = sum(m["xg_self"] * w for m, w in zip(matches, normalized_weights))
    xg_def = sum(m["xg_opp"] * w for m, w in zip(matches, normalized_weights))

    return xg_att, xg_def, weights


def elo_baseline_xg(elo: int) -> float:
    """
    تأثیر مستقیم Elo در xG پایه.
    تیم‌های قوی‌تر xG پایه بالاتری دارند.
    """
    return 1.0 + 0.0005 * (elo - 1500)


def combine_xg(xg_att: float, xg_def_opp: float, alpha: float = 0.65) -> float:
    """
    ترکیب وزن‌دار xG حمله و دفاع حریف.
    
    alpha = 0.65 یعنی ۶۵٪ وزن به xG حمله‌ای و ۳۵٪ به ضعف دفاعی حریف.
    """
    if xg_att <= 0 or xg_def_opp <= 0:
        return (xg_att + xg_def_opp) / 2
    return alpha * xg_att + (1 - alpha) * xg_def_opp


def xg_dominance(att: float, def_: float) -> str:
    """نسبت حمله به دفاع. > 1 یعنی تیم بیشتر می‌سازد تا از دست می‌دهد."""
    return f"{att / def_:.3f}" if def_ > 0 else "N/A"


def team_section(label: str, all_teams: dict):
    st.subheader(label)

    mode = st.radio(
        "روش انتخاب",
        ["از لیست", "دستی"],
        key=f"{label}_mode",
        horizontal=True,
    )
    use_list = mode == "از لیست"

    if use_list:
        team_names = sorted(all_teams.keys())
        selected = st.selectbox("انتخاب تیم", team_names, key=f"{label}_select")
        name = selected
        elo = all_teams[selected]
        st.info(f"رتبه Elo: **{elo}**")
    else:
        name = st.text_input("نام تیم", key=f"{label}_name")
        elo = st.number_input(
            "رتبه Elo", min_value=0, value=1500, step=10, key=f"{label}_elo"
        )

    st.markdown("**۵ بازی آخر**")
    opp_names = sorted(all_teams.keys())
    matches = []

    for i in range(5):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if use_list:
                opp_sel = st.selectbox(
                    f"بازی {i+1} — حریف",
                    opp_names,
                    key=f"{label}_opp_sel_{i}",
                )
                opp_elo = all_teams[opp_sel]
                st.caption(f"Elo: {opp_elo}")
            else:
                opp_elo = st.number_input(
                    f"بازی {i+1} — Elo حریف",
                    min_value=0,
                    value=1500,
                    step=10,
                    key=f"{label}_opp_elo_{i}",
                )
        with c2:
            xg_self = st.number_input(
                f"بازی {i+1} — xG شما",
                min_value=0.0,
                value=1.2,
                step=0.1,
                key=f"{label}_xg_self_{i}",
            )
        with c3:
            xg_opp = st.number_input(
                f"بازی {i+1} — xG حریف",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"{label}_xg_opp_{i}",
            )
        with c4:
            days_ago = st.number_input(
                f"بازی {i+1} — چند روز پیش",
                min_value=0,
                value=i * 7,
                step=1,
                key=f"{label}_days_{i}",
            )
        matches.append({
            "opp_elo": opp_elo,
            "xg_self": xg_self,
            "xg_opp": xg_opp,
            "days_ago": days_ago
        })

    return name, elo, matches


# ── لایه اصلی اپلیکیشن ────────────────────────────────────────────────────────

st.set_page_config(page_title="xG Power Index", layout="wide")
st.title("xG Power Index — مقایسه تیم‌ها")

with st.spinner("در حال بارگذاری رتبه‌های Elo..."):
    all_teams, source = fetch_elo_ratings()

if source == "live":
    st.caption(f"منبع داده Elo: **API زنده** — {len(all_teams)} تیم بارگذاری شد")
else:
    st.caption("منبع داده Elo: **حالت پشتیبان** (اتصال ناموفق)")

col_a, col_b = st.columns(2)
with col_a:
    name_a, elo_a, matches_a = team_section("تیم A", all_teams)
with col_b:
    name_b, elo_b, matches_b = team_section("تیم B", all_teams)

label_a = name_a or "تیم A"
label_b = name_b or "تیم B"

if st.button("تحلیل و پیش‌بینی", type="primary"):
    xg_att_a, xg_def_a, weights_a = analyze_team(elo_a, matches_a)
    xg_att_b, xg_def_b, weights_b = analyze_team(elo_b, matches_b)

    # xG پایه بر اساس Elo
    baseline_a = elo_baseline_xg(elo_a)
    baseline_b = elo_baseline_xg(elo_b)

    # ترکیب xG پایه با xG محاسبه‌شده
    xg_att_a_adjusted = 0.7 * xg_att_a + 0.3 * baseline_a
    xg_att_b_adjusted = 0.7 * xg_att_b + 0.3 * baseline_b

    # xG مورد انتظار برای بازی آینده
    expected_a = combine_xg(xg_att_a_adjusted, xg_def_b)
    expected_b = combine_xg(xg_att_b_adjusted, xg_def_a)

    p_win, p_draw, p_loss = predict_probs(expected_a, expected_b)

    # ── آمار تیم‌ها ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("آمار تیم‌ها (تنظیم‌شده با Elo)")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"**{label_a}**")
        st.metric("xG حمله‌ای (میانگین وزن‌دار)", f"{xg_att_a_adjusted:.3f}")
        st.metric("xG دفاعی خورده (میانگین وزن‌دار)", f"{xg_def_a:.3f}")
        st.metric("تسلط xG (حمله / دفاع)", xg_dominance(xg_att_a_adjusted, xg_def_a))
        st.caption(f"xG پایه از Elo: {baseline_a:.3f}")
    with s2:
        st.markdown(f"**{label_b}**")
        st.metric("xG حمله‌ای (میانگین وزن‌دار)", f"{xg_att_b_adjusted:.3f}")
        st.metric("xG دفاعی خورده (میانگین وزن‌دار)", f"{xg_def_b:.3f}")
        st.metric("تسلط xG (حمله / دفاع)", xg_dominance(xg_att_b_adjusted, xg_def_b))
        st.caption(f"xG پایه از Elo: {baseline_b:.3f}")

    # ── پیش‌بینی بازی ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("پیش‌بینی بازی")
    p1, p2, p3 = st.columns(3)
    p1.metric(f"برد {label_a}", f"{p_win * 100:.1f}%")
    p2.metric("مساوی", f"{p_draw * 100:.1f}%")
    p3.metric(f"برد {label_b}", f"{p_loss * 100:.1f}%")

    if p_win > p_loss and p_win > p_draw:
        st.success(f"پیش‌بینی: **{label_a}** برنده احتمالی است.")
    elif p_loss > p_win and p_loss > p_draw:
        st.success(f"پیش‌بینی: **{label_b}** برنده احتمالی است.")
    else:
        st.info("پیش‌بینی: احتمال **مساوی** زیاد است.")

    # ── گل‌های مورد انتظار ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("گل‌های مورد انتظار برای بازی آینده")
    e1, e2 = st.columns(2)
    e1.metric(f"گل مورد انتظار {label_a}", f"{expected_a:.3f}")
    e2.metric(f"گل مورد انتظار {label_b}", f"{expected_b:.3f}")

    # ── وزن‌های دشواری ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("دشواری بازی‌ها و ضرایب وزن")
    st.caption(
        "وزن = exp((opp_elo − team_elo) / 400) × exp(-0.05 × days_ago). "
        "مقادیر > 1 یعنی حریف قوی‌تر بود."
    )
    w1, w2 = st.columns(2)
    index = [f"بازی {i+1}" for i in range(5)]
    with w1:
        st.markdown(f"**{label_a}** — بالاتر = حریف سخت‌تر")
        st.bar_chart(pd.DataFrame({"ضریب دشواری": weights_a}, index=index))
    with w2:
        st.markdown(f"**{label_b}** — بالاتر = حریف سخت‌تر")
        st.bar_chart(pd.DataFrame({"ضریب دشواری": weights_b}, index=index))
