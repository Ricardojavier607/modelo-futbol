import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import poisson

st.set_page_config(page_title="Modelo Futbol", page_icon="⚽", layout="centered")

ELO_K, ELO_HOME_ADV, ELO_START = 20, 65, 1500
HALF_LIFE_DAYS, RHO_DC, MAX_GOALS = 180, -0.10, 10
KELLY_FRACTION, MAX_STAKE_FRAC = 0.25, 0.02

def implied_probs(oh, od, oa):
    inv = np.array([1/oh, 1/od, 1/oa])
    return inv / inv.sum()

def load_football_data(path_or_file):
    df = pd.read_csv(path_or_file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    keep = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
    df = df.dropna(subset=keep).copy()
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG'] = df['FTAG'].astype(int)
    return df.sort_values('Date').reset_index(drop=True)

def compute_elo(df):
    elo = {}
    for _, r in df.iterrows():
        h, a = r['HomeTeam'], r['AwayTeam']
        eh, ea = elo.get(h, ELO_START), elo.get(a, ELO_START)
        exp_h = 1 / (1 + 10 ** ((ea - (eh + ELO_HOME_ADV)) / 400))
        s_h = {'H':1.0,'D':0.5,'A':0.0}[r['FTR']]
        gd = abs(r['FTHG'] - r['FTAG'])
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = ELO_K * mult * (s_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta
    return elo

def fit_strengths(df, as_of, half_life=HALF_LIFE_DAYS):
    d = df[df['Date'] < as_of]
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
        attack[t]  = (gs / ws) / lg
        defense[t] = (gc / ws) / lg
    return attack, defense, avg_h, avg_a

def predict_lambdas(attack, defense, avg_h, avg_a, home, away):
    return (attack.get(home,1.0) * defense.get(away,1.0) * avg_h,
            attack.get(away,1.0) * defense.get(home,1.0) * avg_a)

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

st.title("⚽ Predictor de Futbol")
st.caption("Poisson + Dixon-Coles + ELO + blend con cuotas + Kelly 1/4")

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("CSV football-data.co.uk", type="csv")
    st.caption("Descarga de football-data.co.uk (E0, SP1, I1, D1, F1...)")

if uploaded is None:
    st.info("Sube un CSV para empezar. Ejemplo: SP1.csv de football-data.co.uk")
    st.stop()

df = load_football_data(uploaded)
st.success(f"{len(df)} partidos cargados ({df['Date'].min().date()} a {df['Date'].max().date()})")

teams = sorted(set(df['HomeTeam']).union(df['AwayTeam']))
home = st.selectbox("Equipo local", teams, index=0)
away = st.selectbox("Equipo visitante", [t for t in teams if t != home], index=0)

use_odds = st.toggle("Usar cuotas del mercado", value=True)
odds = None
if use_odds:
    c1, c2, c3 = st.columns(3)
    oh = c1.number_input("Cuota H", 1.01, 100.0, 2.00, 0.01)
    od = c2.number_input("Cuota D", 1.01, 100.0, 3.30, 0.01)
    oa = c3.number_input("Cuota A", 1.01, 100.0, 3.50, 0.01)
    odds = (oh, od, oa)

bankroll = st.number_input("Bankroll", 1.0, 1e9, 1000.0, 10.0)
blend = st.slider("Peso del modelo en el blend", 0.0, 1.0, 0.5, 0.05)

if st.button("Predecir", use_container_width=True, type="primary"):
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
    c1.metric("Lambda local", f"{lam_h:.2f}")
    c2.metric("Lambda visitante", f"{lam_a:.2f}")

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
        st.markdown("**Stakes (Kelly 1/4, cap 2%)**")
        c1, c2, c3 = st.columns(3)
        for col, tag, p, o in zip([c1, c2, c3], ['H','D','A'], p_final, odds):
            b = o - 1
            edge = p * o - 1
            if edge <= 0:
                col.metric(f"{tag} @ {o:.2f}", "$0", f"edge {edge*100:+.1f}%")
                continue
            kelly = (b * p - (1 - p)) / b
            stake = min(bankroll * kelly * KELLY_FRACTION, bankroll * MAX_STAKE_FRAC)
            col.metric(f"{tag} @ {o:.2f}", f"${stake:.2f}", f"edge {edge*100:+.1f}%")
        st.caption(f"Bankroll: ${bankroll:.0f} · Kelly 1/4 · cap 2% por apuesta")
