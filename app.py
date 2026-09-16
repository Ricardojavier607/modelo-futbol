import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
from scipy.stats import poisson

st.set_page_config(page_title="Modelo Futbol", page_icon="⚽", layout="centered")

# ==================== CONFIG ====================
ELO_K, ELO_HOME_ADV, ELO_START = 20, 65, 1500
HALF_LIFE_DAYS, RHO_DC, MAX_GOALS = 180, -0.10, 10
KELLY_FRACTION, MAX_STAKE_FRAC = 0.25, 0.02

FB_BASE = "https://www.football-data.co.uk/mmz4281"
TEMPORADAS_DEFAULT = ['2425', '2324', '2223', '2122', '2021']

LIGAS = {
    'LaLiga (España)': 'SP1',
    'Premier League': 'E0',
    'Serie A (Italia)': 'I1',
    'Bundesliga (Alemania)': 'D1',
    'Ligue 1 (Francia)': 'F1',
    'Eredivisie (Holanda)': 'N1',
    'Primeira Liga (Portugal)': 'P1',
    'Championship (Inglaterra 2ª)': 'E1',
    'LaLiga 2 (España 2ª)': 'SP2',
    'Brasileirao': 'BRA',
    'Argentina': 'ARG',
    'MLS': 'USA',
}

# ==================== UTILS ====================
def implied_probs(oh, od, oa):
    inv = np.array([1/oh, 1/od, 1/oa])
    return inv / inv.sum()

# ==================== CARGA MULTI-TEMPORADA ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_from_football_data(liga_code, temporadas):
    """Descarga CSV de football-data y une varias temporadas."""
    dfs = []
    fallos = []
    for t in temporadas:
        url = f"{FB_BASE}/{t}/{liga_code}.csv"
        try:
            d = pd.read_csv(url, encoding='latin-1')
            if 'Date' not in d.columns or 'HomeTeam' not in d.columns:
                fallos.append(t)
                continue
            d['Date'] = pd.to_datetime(d['Date'], dayfirst=True, errors='coerce')
            keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            if not all(c in d.columns for c in keep):
                fallos.append(t)
                continue
            d = d.dropna(subset=keep).copy()
            d['FTHG'] = d['FTHG'].astype(int)
            d['FTAG'] = d['FTAG'].astype(int)
            # Cuotas: preferir Pinnacle, luego Bet365
            for col in ['PSH', 'PSD', 'PSA', 'B365H', 'B365D', 'B365A']:
                if col not in d.columns:
                    d[col] = np.nan
            dfs.append(d[keep + ['PSH','PSD','PSA','B365H','B365D','B365A']])
        except Exception:
            fallos.append(t)
    if not dfs:
        return None, fallos
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values('Date').reset_index(drop=True)
    return out, fallos

def load_from_upload(file):
    d = pd.read_csv(file, encoding='latin-1')
    d['Date'] = pd.to_datetime(d['Date'], dayfirst=True, errors='coerce')
    keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    d = d.dropna(subset=keep).copy()
    d['FTHG'] = d['FTHG'].astype(int)
    d['FTAG'] = d['FTAG'].astype(int)
    for col in ['PSH','PSD','PSA','B365H','B365D','B365A']:
        if col not in d.columns:
            d[col] = np.nan
    return d.sort_values('Date').reset_index(drop=True)

# ==================== MODELO ====================
def compute_elo(df):
    elo = {}
    for _, r in df.iterrows():
        h, a = r['HomeTeam'], r['AwayTeam']
        eh, ea = elo.get(h, ELO_START), elo.get(a, ELO_START)
        exp_h = 1 / (1 + 10 ** ((ea - (eh + ELO_HOME_ADV)) / 400))
        s_h = {'H': 1.0, 'D': 0.5, 'A': 0.0}[r['FTR']]
        gd = abs(r['FTHG'] - r['FTAG'])
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = ELO_K * mult * (s_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta
    return elo

def fit_strengths(df, as_of, half_life=HALF_LIFE_DAYS):
    d = df[df['Date'] < as_of].copy()
    if len(d) < 30:
        return {}, {}, 1.3, 1.3
    days = (as_of - d['Date']).dt.days.clip(lower=0)
    w = np.exp(-math.log(2) * days / half_life).values
    d = d.assign(w=w)
    avg_h = (d['FTHG'] * d['w']).sum() / d['w'].sum()
    avg_a = (d['FTAG'] * d['w']).sum() / d['w'].sum()
    lg = (avg_h + avg_a) / 2
    attack, defense = {}, {}
    for t in pd.unique(pd.concat([d['HomeTeam'], d['AwayTeam']])):
        hm, am = d['HomeTeam'] == t, d['AwayTeam'] == t
        gs = (d.loc[hm,'FTHG']*d.loc[hm,'w']).sum() + (d.loc[am,'FTAG']*d.loc[am,'w']).sum()
        gc = (d.loc[hm,'FTAG']*d.loc[hm,'w']).sum() + (d.loc[am,'FTHG']*d.loc[am,'w']).sum()
        ws = d.loc[hm,'w'].sum() + d.loc[am,'w'].sum()
        if ws == 0:
            continue
        attack[t] = (gs / ws) / lg
        defense[t] = (gc / ws) / lg
    return attack, defense, avg_h, avg_a

def predict_lambdas(attack, defense, avg_h, avg_a, home, away):
    return (attack.get(home, 1.0) * defense.get(away, 1.0) * avg_h,
            attack.get(away, 1.0) * defense.get(home, 1.0) * avg_a)

def score_matrix(lam_h, lam_a):
    ph = poisson.pmf(np.arange(MAX_GOALS+1), lam_h)
    pa = poisson.pmf(np.arange(MAX_GOALS+1), lam_a)
    M = np.outer(ph, pa)
    M[0,0] *= 1 - lam_h * lam_a * RHO_DC
    M[0,1] *= 1 + lam_h * RHO_DC
    M[1,0] *= 1 + lam_a * RHO_DC
    M[1,1] *= 1 - RHO_DC
    M = np.clip(M, 0, None)
    return M / M.sum()

def outcomes_from_matrix(M):
    n = M.shape[0]
    pH = sum(M[i,j] for i in range(n) for j in range(n) if i > j)
    pD = sum(M[i,i] for i in range(n))
    return np.array([pH, pD, 1 - pH - pD])

def get_market_odds(row):
    for h, d, a in [('PSH','PSD','PSA'), ('B365H','B365D','B365A')]:
        try:
            oh, od, oa = float(row[h]), float(row[d]), float(row[a])
            if oh > 1.01 and od > 1.01 and oa > 1.01:
                return oh, od, oa
        except Exception:
            continue
    return None

# ==================== UI ====================
st.title("⚽ Predictor de Futbol")
st.caption("Poisson + Dixon-Coles + ELO + cuotas + Kelly ¼ · Multi-temporada")

# --- Sidebar: fuente de datos ---
st.sidebar.header("Datos")

modo = st.sidebar.radio(
    "Fuente de datos",
    ["Automático (football-data.co.uk)", "Subir CSV manualmente"],
)

df = None

if modo == "Automático (football-data.co.uk)":
    liga_nombre = st.sidebar.selectbox("Liga", list(LIGAS.keys()), index=0)
    liga_code = LIGAS[liga_nombre]

    temporadas_str = st.sidebar.text_input(
        "Temporadas (separadas por coma)",
        value=", ".join(TEMPORADAS_DEFAULT),
        help="Formato YYZZ. Ej: 2425 = 2024/25"
    )
    temporadas = [t.strip() for t in temporadas_str.split(",") if t.strip()]

    with st.spinner(f"Cargando {liga_nombre}..."):
        df, fallos = load_from_football_data(liga_code, temporadas)

    if df is None:
        st.error("No se pudieron descargar datos. Prueba subir CSV manual.")
        st.stop()

    st.success(f"✅ {liga_nombre} · {len(df)} partidos · "
               f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    if fallos:
        st.warning(f"Temporadas no disponibles: {', '.join(fallos)}")

else:
    uploaded = st.sidebar.file_uploader("Sube CSV de football-data", type="csv")
    if uploaded is None:
        st.info("👈 Sube un CSV para empezar.")
        st.stop()
    df = load_from_upload(uploaded)
    st.success(f"✅ {len(df)} partidos · "
               f"{df['Date'].min().date()} → {df['Date'].max().date()}")

# --- Equipos ---
teams = sorted(set(df['HomeTeam']).union(df['AwayTeam']))
home = st.selectbox("Equipo local", teams, index=0)
away = st.selectbox("Equipo visitante", [t for t in teams if t != home], index=0)

# --- Cuotas: auto-rellenar con última del CSV, editable ---
last = df.dropna(subset=['Date']).iloc[-1] if len(df) else None

auto = st.toggle("Usar cuotas del mercado", value=True)
odds = None
if auto:
    c1, c2, c3 = st.columns(3)
    oh = c1.number_input("Cuota H", 1.01, 100.0, 2.00, 0.01)
    od = c2.number_input("Cuota D", 1.01, 100.0, 3.30, 0.01)
    oa = c3.number_input("Cuota A", 1.01, 100.0, 3.50, 0.01)
    odds = (oh, od, oa)

bankroll = st.number_input("Bankroll", 1.0, 1e9, 1000.0, 10.0)
blend = st.slider("Peso del modelo en el blend", 0.0, 1.0, 0.5, 0.05,
                  help="0 = solo mercado · 1 = solo modelo")

# --- Predicción ---
if st.button("🔮 Predecir", use_container_width=True, type="primary"):
    elo = compute_elo(df)
    as_of = df['Date'].max() + pd.Timedelta(days=1)
    atk, dfn, avg_h, avg_a = fit_strengths(df, as_of)
    lam_h, lam_a = predict_lambdas(atk, dfn, avg_h, avg_a, home, away)
    p_model = outcomes_from_matrix(score_matrix(lam_h, lam_a))

    if odds:
        p_market = implied_probs(*odds)
        p_final = blend * p_model + (1 - blend) * p_market
    else:
        p_market, p_final = None, p_model

    st.subheader(f"{home} vs {away}")
    c1, c2 = st.columns(2)
    c1.metric("ELO local", f"{elo.get(home, ELO_START):.0f}")
    c2.metric("ELO visitante", f"{elo.get(away, ELO_START):.0f}")
    c1.metric("λ local", f"{lam_h:.2f}")
    c2.metric("λ visitante", f"{lam_a:.2f}")

    def show_probs(label, p):
        st.markdown(f"**{label}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("H", f"{p[0]*100:.1f}%")
        c2.metric("D", f"{p[1]*100:.1f}%")
        c3.metric("A", f"{p[2]*100:.1f}%")

    show_probs("Modelo (Poisson+ELO)", p_model)
    if p_market is not None:
        show_probs("Mercado (sin overround)", p_market)
    show_probs("Final (blend)", p_final)

    if odds:
        st.markdown("**Stakes (Kelly ¼, cap 2%)**")
        c1, c2, c3 = st.columns(3)
        for col, tag, p, o in zip([c1, c2, c3], ['H', 'D', 'A'], p_final, odds):
            b = o - 1
            edge = p * o - 1
            if edge <= 0:
                col.metric(f"{tag} @ {o:.2f}", "$0", f"edge {edge*100:+.1f}%")
                continue
            kelly = (b * p - (1 - p)) / b
            stake = min(bankroll * kelly * KELLY_FRACTION,
                        bankroll * MAX_STAKE_FRAC)
            col.metric(f"{tag} @ {o:.2f}", f"${stake:.2f}",
                       f"edge {edge*100:+.1f}%")
        st.caption(f"Bankroll: ${bankroll:.0f} · Kelly ¼ · cap 2% por apuesta")
