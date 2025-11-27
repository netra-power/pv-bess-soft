import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="BESS Revenue Calculator — PV vs Conso", layout="wide")

# --- Color palette ---
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

st.markdown("<h2 style='text-align: center; color: %s;'>BESS Revenue Calculator</h2>" % COLORS["text"], unsafe_allow_html=True)
st.write("Phase 1 — Superposition des courbes de production photovoltaïque et de consommation")

# ---------------- Sidebar: all inputs and CSV uploads ----------------
st.sidebar.header("Import & Paramètres")

st.sidebar.subheader("📥 Photovoltaïque (PV)")
st.sidebar.write("Importer un fichier CSV de production (optionnel)")
st.sidebar.info("""**Format CSV PV attendu :**
- Séparateur `;`
- Ligne 1 : (DateHeure ; Valeur)
- Ligne 2 : unité dans la 2ème colonne -> (kW) ou (kWh)
- Puis les données : `dd.mm.yyyy HH:MM ; valeur`""")
pv_file = st.sidebar.file_uploader("Importer un extract de production (CSV)", type=["csv"], key="pv_csv")

st.sidebar.markdown("**Sinon : définir la puissance (kWc)**")
pv_kwc = st.sidebar.number_input("Puissance installée (kWc)", min_value=0.0, value=125.0, step=1.0, format="%.1f")
orientation = st.sidebar.selectbox("Orientation", ["Sud", "Est-Ouest"], index=0)
inclinaison = st.sidebar.number_input("Inclinaison (°)", min_value=0, max_value=90, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🏢 Bâtiment — Consommation")
st.sidebar.write("Importer un fichier CSV de consommation (optionnel)")
st.sidebar.info("""**Format CSV consommation attendu :**
- Séparateur `;`
- Ligne 1 : (DateHeure ; Valeur)
- Ligne 2 : unité -> (kW) ou (kWh)
- Puis les données : `dd.mm.yyyy HH:MM ; valeur`""")
cons_file = st.sidebar.file_uploader("Importer un profil de consommation (CSV)", type=["csv"], key="cons_csv")

st.sidebar.markdown("**Sinon : créer un profil type**")
building_type = st.sidebar.selectbox("Type de bâtiment", ["Résidentiel", "Tertiaire", "Industriel"], index=0)
annual_consumption_mwh = st.sidebar.number_input("Consommation annuelle du bâtiment (MWh)", min_value=0.0, value=0.67, step=0.01, format="%.2f")
st.sidebar.caption("La consommation saisie sert à dimensionner le profil type généré automatiquement.")

# ---------------- Helpers and templates ----------------
def make_year_index():
    start = datetime(2001, 1, 1, 0, 0)  # arbitrary non-leap year
    periods = 365 * 24
    return pd.date_range(start, periods=periods, freq="H")

YEAR_IDX = make_year_index()

def pv_annual_profile_south_from_csv(series_125kw):
    s = series_125kw.copy().astype(float)
    s = s.resample("H").mean().reindex(YEAR_IDX, method="nearest", fill_value=0)
    if s.max() == 0:
        return pd.Series(0.0, index=YEAR_IDX)
    return (s / 125.0)

def pv_annual_profile_ew_from_csv(series_125kw):
    s = series_125kw.copy().astype(float)
    s = s.resample("H").mean().reindex(YEAR_IDX, method="nearest", fill_value=0)
    if s.max() == 0:
        return pd.Series(0.0, index=YEAR_IDX)
    return (s / 125.0)

local_south_path = Path("pvsyst_125_sud.CSV")
local_ew_path = Path("pvsyst_125_est_ouest.CSV")
local_south_series = None
local_ew_series = None

def try_load_local(path):
    try:
        if path.exists():
            df = pd.read_csv(path, sep=";", header=None, engine="python", encoding="utf-8")
            if df.shape[1] >= 2:
                col0 = df.iloc[:,0].astype(str)
                col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")
                mask = col1.notna()
                if mask.any():
                    start = mask.idxmax()
                    df2 = pd.DataFrame({"date": col0[start:].values, "val": col1[start:].values})
                else:
                    df2 = pd.DataFrame({"date": col0.values, "val": col1.values})
                df2["date_parsed"] = pd.to_datetime(df2["date"], dayfirst=True, errors="coerce")
                df2 = df2[df2["date_parsed"].notna()]
                if df2.empty:
                    return None
                series = df2.set_index("date_parsed")["val"].resample("H").mean().sort_index()
                mdh = series.index.strftime("%m-%d %H:%M")
                mapping = series.groupby(mdh).median()
                idx_mdh = YEAR_IDX.strftime("%m-%d %H:%M")
                vals = [mapping.get(k, 0.0) for k in idx_mdh]
                return pd.Series(vals, index=YEAR_IDX)
    except Exception:
        return None
    return None

local_south_series = try_load_local(local_south_path)
local_ew_series = try_load_local(local_ew_path)

def pv_south_synthetic():
    idx = YEAR_IDX
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values/60.0
    seasonal = 0.5 + 0.5 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    daily = np.maximum(0, np.cos((hour - 12)/12 * np.pi))
    prod = daily * seasonal
    prod = prod / prod.max() if prod.max()>0 else prod
    return pd.Series(prod, index=idx)

def pv_ew_synthetic():
    idx = YEAR_IDX
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values/60.0
    seasonal = 0.5 + 0.5 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    morning = np.maximum(0, np.cos((hour - 9)/6 * np.pi))
    afternoon = np.maximum(0, np.cos((hour - 15)/6 * np.pi))
    prod = (morning + afternoon) * 0.6 * seasonal
    prod = prod / prod.max() if prod.max()>0 else prod
    return pd.Series(prod, index=idx)

pv_south_template = local_south_series if local_south_series is not None else pv_south_synthetic()
pv_ew_template = local_ew_series if local_ew_series is not None else pv_ew_synthetic()

def load_profile_template(building="Résidentiel"):
    h = np.arange(24)
    if building == "Résidentiel":
        profile = 0.6 + 0.4 * np.exp(-((h - 7) ** 2) / 10) + 0.6 * np.exp(-((h - 19) ** 2) / 12)
    elif building == "Tertiaire":
        profile = 0.4 + 1.2 * np.exp(-((h - 13) ** 2) / 18) * (h >= 7) * (h <= 19)
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
    seasonal = 1.0 + 0.1 * np.cos((doy - 200) / 365 * 2 * np.pi) if building == "Résidentiel" else 1.0
    annual_profile = daily_profile * seasonal
    current_sum = annual_profile.sum()
    factor = total_kwh / current_sum if current_sum>0 else 0.0
    hourly_kw = annual_profile * factor
    return pd.Series(hourly_kw, index=YEAR_IDX)

def parse_csv_timevalue(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=";", header=None, engine="python", encoding="utf-8")
        if df.shape[1] >= 2:
            col0 = df.iloc[:,0].astype(str)
            col1 = pd.to_numeric(df.iloc[:,1], errors="coerce")
            mask_numeric = col1.notna()
            if mask_numeric.any():
                first_idx = mask_numeric.idxmax()
                col0 = col0[first_idx:]
                col1 = col1[first_idx:]
                temp = pd.DataFrame({"date": col0.values, "value": col1.values})
            else:
                temp = pd.DataFrame({"date": col0.values, "value": col1.values})
        else:
            st.warning("Le CSV ne semble pas avoir au moins deux colonnes.")
            return None
        def try_parse(s):
            for fmt in ("%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    pass
            try:
                return pd.to_datetime(s, dayfirst=True, errors="coerce")
            except Exception:
                return pd.NaT
        temp["date_parsed"] = temp["date"].apply(lambda x: try_parse(str(x)))
        temp = temp[temp["date_parsed"].notna()]
        temp = temp.dropna(subset=["value"])
        if temp.empty:
            st.warning("Aucune donnée exploitable trouvée dans le CSV.")
            return None
        temp = temp.set_index(pd.to_datetime(temp["date_parsed"])).sort_index()
        series = temp["value"].resample("H").mean().reindex(YEAR_IDX, method="nearest", fill_value=0)
        return series
    except Exception as e:
        st.warning(f"Erreur lecture CSV : {e}")
        return None

def build_pv_timeseries(pv_file, kwc, orientation, inclinaison):
    if pv_file is not None:
        series = parse_csv_timevalue(pv_file)
        if series is not None:
            df = series.to_frame("val").reset_index()
            df["mdh"] = df["index"].dt.strftime("%m-%d %H:%M")
            mapping = df.groupby("mdh")["val"].median()
            idx_mdh = YEAR_IDX.strftime("%m-%d %H:%M")
            vals = [mapping.get(k, 0.0) for k in idx_mdh]
            return pd.Series(vals, index=YEAR_IDX)
        else:
            st.info("Impossible de parser le CSV PV; utilisation du modèle de référence.")
    if orientation == "Sud":
        base = pv_south_template.copy()
    else:
        base = pv_ew_template.copy()
    series = base * kwc
    if inclinaison < 7:
        series = series * 0.95
    elif inclinaison > 35:
        series = series * 0.93
    return series

def build_consumption_timeseries(cons_file, building_type, annual_mwh):
    if cons_file is not None:
        series = parse_csv_timevalue(cons_file)
        if series is not None:
            df = series.to_frame("val").reset_index()
            df["mdh"] = df["index"].dt.strftime("%m-%d %H:%M")
            mapping = df.groupby("mdh")["val"].median()
            idx_mdh = YEAR_IDX.strftime("%m-%d %H:%M")
            vals = [mapping.get(k, 0.0) for k in idx_mdh]
            return pd.Series(vals, index=YEAR_IDX)
        else:
            st.info("Impossible de parser le CSV consommation; utilisation du modèle synthétique.")
    return make_annual_load_from_profile(building_type, annual_mwh)

pv_ts = build_pv_timeseries(pv_file, pv_kwc, orientation, inclinaison)
cons_ts = build_consumption_timeseries(cons_file, building_type, annual_consumption_mwh)

st.markdown("---")
main_col, side_col = st.columns([2.5, 1])

with main_col:
    st.subheader("Superposition production PV vs consommation (24h)")
    sel_date = st.date_input("Choisir jour et mois (année ignorée)", value=datetime(2001,6,21), key="sel_date")
    month = sel_date.month
    day = sel_date.day
    def extract_day(series, month, day):
        base_idx = pd.date_range(datetime(2001, month, day, 0, 0), periods=24, freq="H")
        mdh_series = series.copy()
        mdh_series.index = mdh_series.index.strftime("%m-%d %H:%M")
        keys = base_idx.strftime("%m-%d %H:%M")
        vals = [mdh_series.get(k, 0.0) for k in keys]
        return pd.Series(vals, index=base_idx)
    pv_day = extract_day(pv_ts, month, day)
    cons_day = extract_day(cons_ts, month, day)

    self_consumed = np.minimum(pv_day, cons_day)
    exported = np.maximum(0, pv_day - cons_day)
    imported = np.maximum(0, cons_day - pv_day)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pv_day.index.hour, y=pv_day.values, name="Production PV (kW)", line=dict(color=COLORS["pv"]), fill='tozeroy'))
    fig.add_trace(go.Scatter(x=cons_day.index.hour, y=cons_day.values, name="Consommation (kW)", line=dict(color=COLORS["load"])))
    fig.update_layout(xaxis_title="Heure (h)", yaxis_title="Puissance (kW)", legend=dict(orientation="h"), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with side_col:
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
    pie1 = px.pie(values=[E_self, E_export], names=["Autoconsommée", "Exportée"], color_discrete_sequence=[COLORS["pv"], COLORS["grid_export"]])
    pie1.update_traces(textinfo='percent+label')
    st.plotly_chart(pie1, use_container_width=True)

    st.markdown("### Source d'énergie pour la consommation (24h)")
    pie2 = px.pie(values=[E_self, E_import], names=["PV (auto)", "Réseau (import)"], color_discrete_sequence=[COLORS["pv"], COLORS["grid_import"]])
    pie2.update_traces(textinfo='percent+label')
    st.plotly_chart(pie2, use_container_width=True)

st.markdown("---")
st.write("Notes :")
st.write("- Les années sont synthétiques : seules la date (jour+mois) et l'heure sont utilisées.")
st.write("- Les CSV importés sont traités en ignorant l'année : seules les combinaisons jour+mois+heure sont retrouvées.")
st.write("- À ce stade la BESS n'est pas encore modélisée : les valeurs de charge/décharge sont à zéro (placeholder).")
st.write("- Les modèles PV utilisés pour l'approximation (Sud / Est-Ouest) utilisent vos fichiers `pvsyst_125_sud.CSV` et `pvsyst_125_est_ouest.CSV` s'ils sont présents dans le dépôt. Sinon des modèles synthétiques sont utilisés.")