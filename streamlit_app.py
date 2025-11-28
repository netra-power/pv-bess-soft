# streamlit_app.py
# Phase 1 — Superposition PV vs Consommation
# Remplacer entièrement le fichier existant par ce contenu.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="BESS Revenue Calculator", layout="wide", initial_sidebar_state="expanded")

# --- Colors ---
COLORS = {
    "pv": "#FFEE8C",
    "bess_charge": "#BFE3F1",
    "bess_discharge": "#7ABAD6",
    "load": "#FFAB72",
    "grid_export": "#D7F4C2",
    "grid_import": "#A8D79E",
    "text": "#2F3A4A",
    "grey": "#9AA0A6",
}

# --- Custom CSS to approximate the pixel look ---
st.markdown(
    f"""
    <style>
    /* Page title style */
    .app-title {{
        font-size: 26px;
        font-weight: 700;
        color: {COLORS['text']};
        margin-bottom: 0.2rem;
    }}
    /* Styled info box like the screenshot */
    .csv-box {{
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.06);
        padding: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,249,252,0.95));
        box-shadow: 0 1px 3px rgba(31,45,61,0.03);
        margin-bottom: 12px;
    }}
    .csv-title {{
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .csv-hint {{
        color: #6b7280;
        font-size: 13px;
        line-height: 1.35;
        margin-bottom: 8px;
    }}
    .blue-note {{
        background: #e8f1fa;
        border-radius: 8px;
        padding: 10px;
        color: #0b5fa5;
        margin-top: 8px;
        margin-bottom: 8px;
    }}
    /* Narrow the sidebar components spacing */
    .sidebar .stButton>button, .sidebar .stNumberInput>div, .sidebar .stSelectbox>div {{
        margin-bottom: 8px;
    }}
    /* minimal padding for date input label */
    .date-label {{ margin-bottom: 6px; }}
    /* keep plot margins comfortable */
    .plotly-graph-div .main-svg {{ background: white; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown(f"<div class='app-title'>Superposition production PV vs consommation (24h)</div>", unsafe_allow_html=True)
st.write("Choisir un jour et un mois (année ignorée) — visualisation sur 24 heures")

# ---------------- Sidebar: CSV + parameters (pixel-like) ----------------
with st.sidebar:
    st.header("Bâtiment — Consommation")
    # Consumption CSV upload box styled with HTML wrapper for pixel look
    st.markdown("<div class='csv-box'>", unsafe_allow_html=True)
    st.markdown("<div class='csv-title'>📥 Importer un fichier CSV de consommation (optionnel)</div>", unsafe_allow_html=True)
    st.markdown("<div class='csv-hint'>Format CSV consommation attendu :</div>", unsafe_allow_html=True)
    st.markdown("<ul style='margin-top:0.2rem; margin-left:18px; margin-bottom:6px; color:#6b7280;'>"
                "<li>Séparateur : <code>;</code></li>"
                "<li>Ligne 1 : (DateHeure ; Valeur)</li>"
                "<li>Ligne 2 : unité dans la 2ème colonne → (kW) ou (kWh)</li>"
                "<li>Données : <code>dd.mm.yyyy HH:MM ; valeur</code></li>"
                "</ul>", unsafe_allow_html=True)
    cons_file = st.file_uploader("Drag and drop file here", type=["csv"], key="cons_csv")
    st.markdown("<div class='blue-note'>Aucun fichier importé — saisissez un profil type ci-dessous.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Profil de consommation du bâtiment")
    building_type = st.selectbox("Type de bâtiment", ["Résidentiel", "Tertiaire", "Industriel"], index=0)
    annual_consumption_mwh = st.number_input("Consommation annuelle (MWh)", min_value=0.0, value=0.67, step=0.01, format="%.2f")
    st.caption("La consommation saisie sert à dimensionner le profil type généré automatiquement.")

    st.markdown("---")
    st.header("Photovoltaïque")
    st.markdown("<div class='csv-box'>", unsafe_allow_html=True)
    st.markdown("<div class='csv-title'>☀️ Importer un fichier CSV de production (optionnel)</div>", unsafe_allow_html=True)
    st.markdown("<div class='csv-hint'>Format CSV PV attendu :</div>", unsafe_allow_html=True)
    st.markdown("<ul style='margin-top:0.2rem; margin-left:18px; margin-bottom:6px; color:#6b7280;'>"
                "<li>Séparateur : <code>;</code></li>"
                "<li>Ligne 1 : (DateHeure ; Valeur)</li>"
                "<li>Ligne 2 : unité dans la 2ème colonne → (kW) ou (kWh)</li>"
                "<li>Données : <code>dd.mm.yyyy HH:MM ; valeur</code></li>"
                "</ul>", unsafe_allow_html=True)
    pv_file = st.file_uploader("Drag and drop file here", type=["csv"], key="pv_csv")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Sinon : définir la puissance (kWc)**")
    pv_kwc = st.number_input("Puissance installée (kWc)", min_value=0.0, value=125.0, step=1.0, format="%.1f")
    orientation = st.selectbox("Orientation", ["Sud", "Est-Ouest"], index=0)
    inclinaison = st.number_input("Inclinaison (°)", min_value=0, max_value=90, value=20, step=1)

    st.markdown("---")
    st.write("Fichiers PV templates (optionnels) :")
    st.write("- `pvsyst_125_sud.CSV` et `pvsyst_125_est_ouest.CSV` à la racine du dépôt (chargés automatiquement si présents).")

# ------------------ Helpers ------------------
def make_year_index():
    start = datetime(2001, 1, 1, 0, 0)  # template non-leap year
    return pd.date_range(start, periods=365*24, freq="H")

YEAR_IDX = make_year_index()

def gaussian_smooth(series, sigma_hours=2.0):
    """Apply gaussian smoothing using convolution (sigma in hours)."""
    if getattr(series, "empty", False):
        return series
    window_radius = max(1, int(sigma_hours * 3))
    x = np.arange(-window_radius, window_radius+1)
    sigma = sigma_hours
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    vals = np.convolve(series.values, kernel, mode="same")
    return pd.Series(vals, index=series.index)

# ------------------ CSV parsing (robust) ------------------
def parse_csv_timevalue(uploaded, sep=";"):
    """uploaded: file-like or path string. returns hourly Series indexed to YEAR_IDX or None."""
    try:
        if isinstance(uploaded, (str, Path)):
            df = pd.read_csv(str(uploaded), sep=sep, header=None, engine="python", encoding="utf-8")
        else:
            df = pd.read_csv(uploaded, sep=sep, header=None, engine="python", encoding="utf-8")
    except Exception as e:
        st.warning(f"Erreur lecture CSV: {e}")
        return None

    if df.shape[1] < 2:
        st.warning("Le CSV ne contient pas au moins deux colonnes.")
        return None
    col0 = df.iloc[:,0].astype(str)
    col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")
    mask = col1.notna()
    if mask.any():
        start = mask.idxmax()
        df2 = pd.DataFrame({"date": col0[start:].values, "val": col1[start:].values})
    else:
        df2 = pd.DataFrame({"date": col0.values, "val": col1.values})

    # parse dates (try multiple formats)
    def try_parse(s):
        for fmt in ("%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        # try pandas fallback
        try:
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        except Exception:
            return pd.NaT

    df2["date_parsed"] = df2["date"].apply(lambda x: try_parse(str(x)))
    df2 = df2[df2["date_parsed"].notna()]
    df2 = df2.dropna(subset=["val"])
    if df2.empty:
        st.info("Aucune donnée exploitable trouvée dans le CSV.")
        return None

    series = df2.set_index("date_parsed")["val"].sort_index().resample("H").mean()
    # map to template YEAR_IDX by mdh
    mdh = series.index.strftime("%m-%d %H:%M")
    mapping = series.groupby(mdh).median()
    idx_mdh = YEAR_IDX.strftime("%m-%d %H:%M")
    vals = [mapping.get(k, 0.0) for k in idx_mdh]
    return pd.Series(vals, index=YEAR_IDX)

# ------------------ Load local templates if present ------------------
def try_load_local_template(path):
    path = Path(path)
    if not path.exists():
        return None
    s = parse_csv_timevalue(str(path))
    return s

local_sud = try_load_local_template("pvsyst_125_sud.CSV")
local_ew = try_load_local_template("pvsyst_125_est_ouest.CSV")

# If no local template, create synthetic templates
def pv_south_synthetic():
    idx = YEAR_IDX
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values/60.0
    seasonal = 0.45 + 0.55 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    daily = np.maximum(0, np.cos((hour - 12)/12 * np.pi))
    prod = daily * seasonal
    if prod.max() > 0:
        prod = prod / prod.max() * 125.0  # keep 125 max for template
    return pd.Series(prod, index=idx)

def pv_ew_synthetic():
    idx = YEAR_IDX
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values/60.0
    seasonal = 0.45 + 0.55 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    morning = np.maximum(0, np.cos((hour - 9)/6 * np.pi))
    afternoon = np.maximum(0, np.cos((hour - 15)/6 * np.pi))
    prod = (morning + afternoon) * 0.6 * seasonal
    if prod.max() > 0:
        prod = prod / prod.max() * 125.0
    return pd.Series(prod, index=idx)

pv_south_template = local_sud if local_sud is not None else pv_south_synthetic()
pv_ew_template = local_ew if local_ew is not None else pv_ew_synthetic()

# Smooth templates (reduce artificial spikes)
pv_south_template = gaussian_smooth(pv_south_template, sigma_hours=2.5)
pv_ew_template = gaussian_smooth(pv_ew_template, sigma_hours=2.5)

# ------------------ Build PV timeseries (kW hourly) ------------------
def build_pv_timeseries(pv_file, kwc, orientation, inclinaison):
    if pv_file is not None:
        s = parse_csv_timevalue(pv_file)
        if s is not None:
            s = gaussian_smooth(s, sigma_hours=2.0)
            # if CSV provided in kWh per hour, treat as kW average over hour
            return s
        else:
            st.info("Impossible de parser le CSV PV ; utilisation du modèle template.")
    # use template normalized per 125 kW and scale to kwc
    if orientation == "Sud":
        base = pv_south_template.copy()
    else:
        base = pv_ew_template.copy()
    # template currently has magnitude ~125 peak; normalize per 1 kWc and multiply by kwc
    # avoid division by zero
    max_base = base.max() if base.max() > 0 else 1.0
    per_kw = base / (max_base)  # normalized to 1 at peak
    # scale so peak equals kwc * (approx peak/peak) -> use per_kw * kwc * (max_base/125) to retain absolute shape
    # simpler: scale as per_kw * kwc * (max_base/1) -> acceptable
    series = per_kw * kwc * (max_base / max_base)  # effectively per_kw * kwc
    # tilt correction: -5% if inclinaison < 7°, 0% for 7-35°, -7% if >35°
    if inclinaison < 7:
        series = series * 0.95
    elif inclinaison > 35:
        series = series * 0.93
    # smooth final
    series = gaussian_smooth(series, sigma_hours=1.8)
    return series

# ------------------ Build consumption timeseries ------------------
def load_profile_template(building="Résidentiel"):
    h = np.arange(24)
    if building == "Résidentiel":
        profile = 0.6 + 0.4 * np.exp(-((h - 7) ** 2) / 10) + 0.6 * np.exp(-((h - 19) ** 2) / 12)
    elif building == "Tertiaire":
        profile = 0.4 + 1.2 * np.exp(-((h - 13) ** 2) / 18) * ((h >= 7) & (h <= 19))
        profile = profile + 0.1
    else:
        profile = 1.0 + 0.2 * np.exp(-((h - 12) ** 2) / 60)
    profile = profile / np.mean(profile)
    return profile

def make_annual_load_from_profile(building, annual_mwh):
    total_kwh = annual_mwh * 1000.0
    template_daily = load_profile_template(building)
    daily_profile = np.tile(template_daily, 365)
    doy = YEAR_IDX.dayofyear.values
    seasonal = 1.0 + 0.08 * np.cos((doy - 200) / 365 * 2 * np.pi) if building == "Résidentiel" else 1.0
    annual_profile = daily_profile * seasonal
    current_sum = annual_profile.sum()
    factor = total_kwh / current_sum if current_sum > 0 else 0.0
    hourly_kw = annual_profile * factor
    return pd.Series(hourly_kw, index=YEAR_IDX)

def build_consumption_timeseries(cons_file, building_type, annual_mwh):
    if cons_file is not None:
        s = parse_csv_timevalue(cons_file)
        if s is not None:
            s = gaussian_smooth(s, sigma_hours=1.0)
            return s
        else:
            st.info("Impossible de parser le CSV consommation ; utilisation du modèle synthétique.")
    return make_annual_load_from_profile(building_type, annual_mwh)

# ------------------ Build series ------------------
pv_ts = build_pv_timeseries(pv_file, pv_kwc, orientation, inclinaison)
cons_ts = build_consumption_timeseries(cons_file, building_type, annual_consumption_mwh)

# ------------------ Main layout: graph & controls ------------------
st.markdown("---")
left, right = st.columns([2.6, 1])

# Date input shown above graph area (but placed inside left column to be visually close)
with left:
    sel_date = st.date_input("Choisir jour et mois (année ignorée)", value=datetime(2001, 6, 21), key="sel_date")
    month = sel_date.month
    day = sel_date.day

    # extract 24 hours by month/day mapping
    def extract_day(series, month, day):
        base_idx = pd.date_range(datetime(2001, month, day, 0, 0), periods=24, freq="H")
        mdh_series = series.copy()
        mdh_series.index = mdh_series.index.strftime("%m-%d %H:%M")
        keys = base_idx.strftime("%m-%d %H:%M")
        vals = [mdh_series.get(k, 0.0) for k in keys]
        return pd.Series(vals, index=base_idx)

    pv_day = extract_day(pv_ts, month, day)
    cons_day = extract_day(cons_ts, month, day)

    # energy flows (simple, no BESS)
    self_consumed = np.minimum(pv_day, cons_day)
    exported = np.maximum(0, pv_day - cons_day)
    imported = np.maximum(0, cons_day - pv_day)

    # plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pv_day.index.hour, y=pv_day.values, name="Production PV (kW)",
                             line=dict(color=COLORS["pv"], width=2), fill='tozeroy', fillcolor=COLORS["pv"]))
    fig.add_trace(go.Scatter(x=cons_day.index.hour, y=cons_day.values, name="Consommation (kW)",
                             line=dict(color=COLORS["load"], width=2)))
    fig.update_layout(
        xaxis=dict(title="Heure (h)", tickmode='linear', dtick=1),
        yaxis=dict(title="Puissance (kW)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("Indicateurs — 24h sélectionnées")
    total_pv = pv_day.sum()
    total_load = cons_day.sum()
    E_self = self_consumed.sum()
    E_export = exported.sum()
    E_import = imported.sum()

    st.metric("Production PV totale (kWh)", f"{total_pv:.1f}")
    st.metric("Consommation totale (kWh)", f"{total_load:.1f}")
    st.metric("Autoconsommation (kWh)", f"{E_self:.1f}")

    st.markdown("### Répartition de l'énergie (PV)")
    pie1 = px.pie(values=[E_self, E_export], names=["Autoconsommée", "Exportée"],
                  color_discrete_sequence=[COLORS["pv"], COLORS["grid_export"]])
    pie1.update_traces(textinfo='percent+label')
    st.plotly_chart(pie1, use_container_width=True, height=260)

    st.markdown("### Source d'énergie pour la consommation (24h)")
    pie2 = px.pie(values=[E_self, E_import], names=["PV (auto)", "Réseau (import)"],
                  color_discrete_sequence=[COLORS["pv"], COLORS["grid_import"]])
    pie2.update_traces(textinfo='percent+label')
    st.plotly_chart(pie2, use_container_width=True, height=260)

st.markdown("---")
st.write("Notes :")
st.write("- Les années sont synthétiques : on n'utilise que le jour+mois+heure pour comparer.")
st.write("- Les CSV importés sont interprétés en ignorant l'année : seules les combinaisons jour+mois+heure sont utilisées.")
st.write("- Tilt correction : inclinaison < 7° → -5% ; 7°–35° → 0% ; >35° → -7%")
st.write("- Les templates PV (125 kW) sont chargés depuis `pvsyst_125_sud.CSV` et `pvsyst_125_est_ouest.CSV` si présents.")

# End of file
