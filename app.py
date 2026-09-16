import streamlit as st
import pandas as pd
import numpy as np
import math
import altair as alt
from datetime import datetime
from scipy.stats import poisson

st.set_page_config(page_title="Modelo Futbol", page_icon="⚽", layout="wide")

# ==================== CONFIG ====================
ELO_K, ELO_HOME_ADV, ELO_START = 20, 65, 1500
HALF_LIFE_DAYS, RHO_DC, MAX_GOALS = 180, -0.10, 10
KELLY_FRACTION, MAX_STAKE_FRAC = 0.25, 0.02
MIN_MATCHES_WARN = 5
BACKTEST_WINDOW = 500

FB_BASE = "https://www.football-data.co.uk/mmz4281"
TEMPORADAS_DEFAULT = ['2425', '2324', '2223', '2122', '2021']

ODDS_CANDIDATES = [
    ('PSCH','PSCD','PSCA'),   # Pinnacle cierre
    ('PSH','PSD','PSA'),      # Pinnacle apertura
    ('B365CH','B365CD','B365CA'),
    ('B365H','B365D','B365A'),
    ('WHH','WHD','WHA'),
    ('IWH','IWD','IWA'),
    ('VCH','VCD','VCA'),
    ('BWH','BWD','BWA'),
]

ALL_ODDS_COLS = [
    'PSCH','PSCD','PSCA','PSH','PSD','PSA',
    'B365CH','B365CD','B365CA','B365H','B365D','B365A',
    'WHH','WHD','WHA','IWH','IWD','IWA',
    'VCH','VCD','VCA','BWH','BWD','BWA',
]

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

# ==================== SESSION ====================
if 'predicciones' not in st.session_state:
    st.session_state.predicciones = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'backtest_result' not in st.session_state:
    st.session_state.backtest_result = None

# ==================== UTILS ====================
def implied_probs(oh, od, oa):
    inv = np.array([1/oh, 1/od, 1/oa])
    return inv / inv.sum()

def kelly_stake(p, o, bankroll):
    b = o - 1
    edge = p * o - 1
    if edge <= 0:
        return 0.0, edge
    kelly = (b * p - (1 - p)) / b
    stake = min(bankroll * kelly * KELLY_FRACTION, bankroll * MAX_STAKE_FRAC)
    return max(0.0, stake), edge

def get_odds_from_row(r):
    for h_col, d_col, a_col in ODDS_CANDIDATES:
        if h_col not in r.index:
            continue
        try:
            oh = float(r[h_col]); od = float(r[d_col]); oa = float(r[a_col])
            if oh > 1.01 and od > 1.01 and oa > 1.01:
                return oh, od, oa
        except (TypeError, ValueError, KeyError):
            continue
    return None

# ==================== CARGA ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_from_football_data(liga_code, temporadas):
    dfs, fallos = [], []
    for t in temporadas:
        url = f"{FB_BASE}/{t}/{liga_code}.csv"
        try:
            d = pd.read_csv(url, encoding='latin-1')
            if 'Date' not in d.columns or 'HomeTeam' not in d.columns:
                fallos.append(t); continue
            d['Date'] = pd.to_datetime(d['Date'], dayfirst=True, errors='coerce')
            keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            if not all(c in d.columns for c in keep):
                fallos.append(t); continue
            d = d.dropna(subset=keep).copy()
            d['FTHG'] = d['FTHG'].astype(int)
            d['FTAG'] = d['FTAG'].astype(int)
            for col in ALL_ODDS_COLS:
                if col not in d.columns:
                    d[col] = np.nan
            dfs.append(d[keep + ALL_ODDS_COLS])
        except Exception:
            fallos.append(t)
    if not dfs:
        return None, fallos
    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values('Date').reset_index(drop=True), fallos

def load_from_upload(file):
    d = pd.read_csv(file, encoding='latin-1')
    d['Date'] = pd.to_datetime(d['Date'], dayfirst=True, errors='coerce')
    keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    d = d.dropna(subset=keep).copy()
    d['FTHG'] = d['FTHG'].astype(int)
    d['FTAG'] = d['FTAG'].astype(int)
    for col in ALL_ODDS_COLS:
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

def fit_strengths(df, half_life=HALF_LIFE_DAYS):
    if len(df) < 30:
        return {}, {}, 1.3, 1.3
    as_of = df['Date'].max() + pd.Timedelta(days=1)
    d = df.copy()
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

def extra_markets(M):
    n = M.shape[0]
    over25 = sum(M[i,j] for i in range(n) for j in range(n) if i + j > 2)
    btts = sum(M[i,j] for i in range(1, n) for j in range(1, n))
    all_scores = [(f"{i}-{j}", float(M[i,j]))
                  for i in range(min(5, n)) for j in range(min(5, n))]
    all_scores.sort(key=lambda x: -x[1])
    seen, top = set(), []
    for sc, p in all_scores:
        if sc in seen:
            continue
        seen.add(sc)
        top.append((sc, p))
        if len(top) == 5:
            break
    return {'over_2_5': over25, 'under_2_5': 1 - over25,
            'btts_yes': btts, 'btts_no': 1 - btts,
            'top_scores': top}

# ==================== BACKTEST ====================
def brier_3way(p_h, p_d, p_a, outcome_idx):
    o = [0.0, 0.0, 0.0]
    o[outcome_idx] = 1.0
    return ((p_h - o[0])**2 + (p_d - o[1])**2 + (p_a - o[2])**2) / 2

def logloss_3way(p_h, p_d, p_a, outcome_idx):
    p = [p_h, p_d, p_a][outcome_idx]
    return -math.log(max(p, 1e-12))

def run_backtest(df, blend_w=0.5, sample_n=500, min_history=200, window=BACKTEST_WINDOW):
    df = df.sort_values('Date').reset_index(drop=True)
    n = len(df)
    start = max(min_history, n - sample_n)

    elo = {}
    elo_before = []
    for _, r in df.iterrows():
        h, a = r['HomeTeam'], r['AwayTeam']
        eh, ea = elo.get(h, ELO_START), elo.get(a, ELO_START)
        elo_before.append((eh, ea))
        exp_h = 1 / (1 + 10 ** ((ea - (eh + ELO_HOME_ADV)) / 400))
        s_h = {'H': 1.0, 'D': 0.5, 'A': 0.0}[r['FTR']]
        gd = abs(r['FTHG'] - r['FTAG'])
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = ELO_K * mult * (s_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    rows = []
    for i in range(start, n):
        r = df.iloc[i]
        train = df.iloc[max(0, i - window):i]
        if len(train) < 30:
            continue
        atk, dfn, avg_h, avg_a = fit_strengths(train)
        lam_h, lam_a = predict_lambdas(atk, dfn, avg_h, avg_a, r['HomeTeam'], r['AwayTeam'])
        M = score_matrix(lam_h, lam_a)
        p_model = outcomes_from_matrix(M)

        odds = get_odds_from_row(r)
        if odds:
            p_market = implied_probs(*odds)
            p_final = blend_w * p_model + (1 - blend_w) * p_market
        else:
            p_market = None
            p_final = p_model

        outcome_idx = {'H': 0, 'D': 1, 'A': 2}[r['FTR']]
        rows.append({
            'Date': r['Date'],
            'HomeTeam': r['HomeTeam'], 'AwayTeam': r['AwayTeam'],
            'outcome': outcome_idx,
            'elo_h': elo_before[i][0], 'elo_a': elo_before[i][1],
            'p_model_H': p_model[0], 'p_model_D': p_model[1], 'p_model_A': p_model[2],
            'p_market_H': p_market[0] if p_market is not None else np.nan,
            'p_market_D': p_market[1] if p_market is not None else np.nan,
            'p_market_A': p_market[2] if p_market is not None else np.nan,
            'p_final_H': p_final[0], 'p_final_D': p_final[1], 'p_final_A': p_final[2],
            'oh': odds[0] if odds else np.nan,
            'od': odds[1] if odds else np.nan,
            'oa': odds[2] if odds else np.nan,
        })
    return pd.DataFrame(rows)

def metrics_table(bt):
    out = []
    # Modelo
    if len(bt):
        br = sum(brier_3way(r['p_model_H'], r['p_model_D'], r['p_model_A'], r['outcome'])
                 for _, r in bt.iterrows()) / len(bt)
        ll = sum(logloss_3way(r['p_model_H'], r['p_model_D'], r['p_model_A'], r['outcome'])
                 for _, r in bt.iterrows()) / len(bt)
        out.append(('Modelo', len(bt), br, ll))
    # Mercado
    sub = bt.dropna(subset=['p_market_H'])
    if len(sub):
        br = sum(brier_3way(r['p_market_H'], r['p_market_D'], r['p_market_A'], r['outcome'])
                 for _, r in sub.iterrows()) / len(sub)
        ll = sum(logloss_3way(r['p_market_H'], r['p_market_D'], r['p_market_A'], r['outcome'])
                 for _, r in sub.iterrows()) / len(sub)
        out.append(('Mercado', len(sub), br, ll))
    # Final
    if len(bt):
        br = sum(brier_3way(r['p_final_H'], r['p_final_D'], r['p_final_A'], r['outcome'])
                 for _, r in bt.iterrows()) / len(bt)
        ll = sum(logloss_3way(r['p_final_H'], r['p_final_D'], r['p_final_A'], r['outcome'])
                 for _, r in bt.iterrows()) / len(bt)
        out.append(('Final', len(bt), br, ll))
    # Uniforme
    if len(bt):
        br = sum(brier_3way(1/3, 1/3, 1/3, r['outcome'])
                 for _, r in bt.iterrows()) / len(bt)
        ll = sum(logloss_3way(1/3, 1/3, 1/3, r['outcome'])
                 for _, r in bt.iterrows()) / len(bt)
        out.append(('Uniforme', len(bt), br, ll))
    return pd.DataFrame(out, columns=['Fuente', 'N', 'Brier', 'LogLoss'])

def simulate_betting(bt, bankroll=1000, edge_min=0.02, bet_filter='all',
                     kelly_frac=KELLY_FRACTION, cap=MAX_STAKE_FRAC):
    current = bankroll
    total_staked = 0.0
    n_bets = 0
    n_won = 0
    log = []
    for _, r in bt.iterrows():
        if not (r['oh'] > 1.01 and r['od'] > 1.01 and r['oa'] > 1.01):
            continue
        probs = [r['p_final_H'], r['p_final_D'], r['p_final_A']]
        odds = [r['oh'], r['od'], r['oa']]
        for k, tag in enumerate(['H', 'D', 'A']):
            if bet_filter != 'all' and tag != bet_filter:
                continue
            edge = probs[k] * odds[k] - 1
            if edge < edge_min:
                continue
            b = odds[k] - 1
            kelly = (b * probs[k] - (1 - probs[k])) / b
            stake = min(current * kelly * kelly_frac, current * cap)
            if stake < 0.01:
                continue
            n_bets += 1
            total_staked += stake
            won = (r['outcome'] == k)
            if won:
                n_won += 1
                current += stake * b
            else:
                current -= stake
            log.append({
                'Date': r['Date'], 'Match': f"{r['HomeTeam']} vs {r['AwayTeam']}",
                'Bet': tag, 'Odds': odds[k], 'Edge_%': round(edge*100, 2),
                'Stake': round(stake, 2),
                'Resultado': ['H','D','A'][r['outcome']],
                'Ganó': 'SÍ' if won else 'NO',
                'Bankroll': round(current, 2),
            })
    roi = (current - bankroll) / total_staked if total_staked > 0 else 0
    return {
        'bankroll_start': bankroll, 'bankroll_end': current,
        'pnl': current - bankroll, 'n_bets': n_bets, 'n_won': n_won,
        'hit_rate': n_won / n_bets if n_bets else 0,
        'total_staked': total_staked, 'roi': roi,
    }, pd.DataFrame(log)

# ==================== UI ====================
st.title("⚽ Predictor de Futbol")
st.caption("Poisson + Dixon-Coles + ELO + cuotas + Kelly ¼ · Multi-temporada")

st.sidebar.header("Datos")
modo = st.sidebar.radio("Fuente de datos",
    ["Automático (football-data.co.uk)", "Subir CSV manualmente"])

df = None
liga_nombre = "Manual"

if modo == "Automático (football-data.co.uk)":
    liga_nombre = st.sidebar.selectbox("Liga", list(LIGAS.keys()), index=0)
    liga_code = LIGAS[liga_nombre]
    temporadas_str = st.sidebar.text_input(
        "Temporadas (separadas por coma)",
        value=", ".join(TEMPORADAS_DEFAULT))
    temporadas = [t.strip() for t in temporadas_str.split(",") if t.strip()]
    with st.spinner(f"Cargando {liga_nombre}..."):
        df, fallos = load_from_football_data(liga_code, temporadas)
    if df is None:
        st.error("No se pudieron descargar datos.")
        st.stop()
    st.sidebar.success(f"✅ {len(df)} partidos · {df['Date'].min().date()} → {df['Date'].max().date()}")
    if fallos:
        st.sidebar.warning(f"Faltan: {', '.join(fallos)}")
else:
    uploaded = st.sidebar.file_uploader("Sube CSV", type="csv")
    if uploaded is None:
        st.info("👈 Sube un CSV para empezar.")
        st.stop()
    df = load_from_upload(uploaded)
    st.sidebar.success(f"✅ {len(df)} partidos")

# ==================== TABS ====================
tab_pred, tab_bt = st.tabs(["🔮 Predicción", "📊 Backtest"])

# ==================== TAB 1: PREDICCIÓN ====================
with tab_pred:
    teams = sorted(set(df['HomeTeam']).union(df['AwayTeam']))
    home = st.selectbox("Equipo local", teams, index=0)
    away = st.selectbox("Equipo visitante", [t for t in teams if t != home], index=0)

    auto = st.toggle("Usar cuotas del mercado", value=True)
    odds = None
    o_over = o_under = o_btts = None

    if auto:
        st.markdown("**Cuotas 1X2**")
        c1, c2, c3 = st.columns(3)
        oh = c1.number_input("Cuota H", 1.01, 100.0, 2.00, 0.01)
        od = c2.number_input("Cuota D", 1.01, 100.0, 3.30, 0.01)
        oa = c3.number_input("Cuota A", 1.01, 100.0, 3.50, 0.01)
        odds = (oh, od, oa)

        st.markdown("**Cuotas mercados extra (opcional, 0 = no usar)**")
        c1, c2, c3 = st.columns(3)
        o_over = c1.number_input("Over 2.5", 0.0, 100.0, 0.0, 0.05)
        o_under = c2.number_input("Under 2.5", 0.0, 100.0, 0.0, 0.05)
        o_btts = c3.number_input("BTTS Sí", 0.0, 100.0, 0.0, 0.05)

    bankroll = st.number_input("Bankroll", 1.0, 1e9, 1000.0, 10.0)
    blend = st.slider("Peso del modelo en el blend", 0.0, 1.0, 0.5, 0.05)

    if st.button("🔮 Predecir", use_container_width=True, type="primary"):
        elo = compute_elo(df)
        atk, dfn, avg_h, avg_a = fit_strengths(df)
        lam_h, lam_a = predict_lambdas(atk, dfn, avg_h, avg_a, home, away)
        M = score_matrix(lam_h, lam_a)
        p_model = outcomes_from_matrix(M)
        extras = extra_markets(M)

        if odds:
            p_market = implied_probs(*odds)
            p_final = blend * p_model + (1 - blend) * p_market
        else:
            p_market, p_final = None, p_model

        n_h = int(((df['HomeTeam'] == home) | (df['AwayTeam'] == home)).sum())
        n_a = int(((df['HomeTeam'] == away) | (df['AwayTeam'] == away)).sum())

        st.session_state.last_result = {
            'home': home, 'away': away, 'liga': liga_nombre,
            'elo_home': elo.get(home, ELO_START), 'elo_away': elo.get(away, ELO_START),
            'n_home': n_h, 'n_away': n_a,
            'lam_h': lam_h, 'lam_a': lam_a,
            'p_model': p_model.tolist(),
            'p_market': p_market.tolist() if p_market is not None else None,
            'p_final': p_final.tolist(),
            'extras': extras, 'odds': odds,
            'o_over': o_over if o_over and o_over > 1.01 else None,
            'o_under': o_under if o_under and o_under > 1.01 else None,
            'o_btts': o_btts if o_btts and o_btts > 1.01 else None,
            'bankroll': bankroll,
        }

        rec = {
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'liga': liga_nombre, 'local': home, 'visitante': away,
            'elo_local': round(elo.get(home, ELO_START), 1),
            'elo_visit': round(elo.get(away, ELO_START), 1),
            'lam_local': round(lam_h, 2), 'lam_visit': round(lam_a, 2),
            'p_mod_H': round(p_model[0] * 100, 1),
            'p_mod_D': round(p_model[1] * 100, 1),
            'p_mod_A': round(p_model[2] * 100, 1),
            'p_fin_H': round(p_final[0] * 100, 1),
            'p_fin_D': round(p_final[1] * 100, 1),
            'p_fin_A': round(p_final[2] * 100, 1),
            'over25': round(extras['over_2_5'] * 100, 1),
            'btts': round(extras['btts_yes'] * 100, 1),
        }
        if odds:
            rec.update({'cuota_H': oh, 'cuota_D': od, 'cuota_A': oa})
        st.session_state.predicciones.append(rec)

    if st.session_state.last_result is not None:
        r = st.session_state.last_result
        st.subheader(f"{r['home']} vs {r['away']}")

        if r['n_home'] < MIN_MATCHES_WARN or r['n_away'] < MIN_MATCHES_WARN:
            st.warning(f"⚠️ Pocos datos: {r['home']} {r['n_home']} · {r['away']} {r['n_away']}")

        c1, c2 = st.columns(2)
        c1.metric("ELO local", f"{r['elo_home']:.0f}", f"{r['n_home']} partidos")
        c2.metric("ELO visitante", f"{r['elo_away']:.0f}", f"{r['n_away']} partidos")
        c1.metric("λ local", f"{r['lam_h']:.2f}")
        c2.metric("λ visitante", f"{r['lam_a']:.2f}")

        pm, pk, pf = r['p_model'], r['p_market'], r['p_final']

        def show(label, p):
            st.markdown(f"**{label}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("H", f"{p[0]*100:.1f}%")
            c2.metric("D", f"{p[1]*100:.1f}%")
            c3.metric("A", f"{p[2]*100:.1f}%")

        show("Modelo", pm)
        if pk is not None:
            show("Mercado", pk)
        show("Final (blend)", pf)

        # Gráfico SIN tooltips
        chart_df = pd.DataFrame({
            'Fuente': ['Modelo']*3 + ['Mercado']*3 + ['Final']*3,
            'Resultado': ['H', 'D', 'A'] * 3,
            'Probabilidad': [
                pm[0]*100, pm[1]*100, pm[2]*100,
                (pk[0]*100 if pk else 0), (pk[1]*100 if pk else 0), (pk[2]*100 if pk else 0),
                pf[0]*100, pf[1]*100, pf[2]*100,
            ],
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Resultado:N', axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y('Probabilidad:Q', title='Probabilidad (%)'),
            color=alt.Color('Resultado:N', scale=alt.Scale(
                domain=['H', 'D', 'A'],
                range=['#5B8FF9', '#F6BD16', '#E8684A'])),
   
