import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

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

# --- Left column: Inputs ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 Import / Photovoltaïque")

    st.write("**Importer un fichier CSV de production (optionnel)**")
    st.info("""**Format CSV PV attendu :**
- Séparateur `;`
- Ligne 1 : (DateHeure ; Valeur)
- Ligne 2 : unité dans la 2ème colonne -> (kW) ou (kWh)
- Puis les données : `dd.mm.yyyy HH:MM ; valeur`""")

    pv_file = st.file_uploader("Importer un extract de production (CSV)", type=["csv"], key="pv_csv")
    st.markdown("---")
    st.write("**Sinon : définir la puissance (kWc)**")
    pv_kwc = st.number_input("Puissance installée (kWc)", min_value=0.0, value=100.0, step=1.0, format="%.1f")
    orientation = st.selectbox("Orientation", ["Sud", "Est-Ouest"], index=0)
    inclinaison = st.number_input("Inclinaison (°)", min_value=0, max_value=90, value=20, step=1)

with col2:
    st.subheader("🏢 Bâtiment — Consommation")

    st.write("**Importer un fichier CSV de consommation (optionnel)**")
    st.info("""**Format CSV consommation attendu :**
- Séparateur `;`
- Ligne 1 : (DateHeure ; Valeur)
- Ligne 2 : unité -> (kW) ou (kWh)
- Puis les données : `dd.mm.yyyy HH:MM ; valeur`""")

    cons_file = st.file_uploader("Importer un profil de consommation (CSV)", type=["csv"], key="cons_csv")

    st.markdown("---")
    st.write("**Sinon : créer un profil type**")
    building_type = st.selectbox("Type de bâtiment", ["Résidentiel", "Tertiaire", "Industriel"], index=0)
    annual_consumption_mwh = st.number_input("Consommation annuelle du bâtiment (MWh)", min_value=0.0, value=0.67, step=0.01, format="%.2f")
    st.caption("La consommation saisie sert à dimensionner le profil type généré automatiquement.")

st.markdown("---")

# --- Helpers: time index (one non-leap year) ---
def make_year_index():
    start = datetime(2001, 1, 1, 0, 0)  # arbitrary non-leap year
    periods = 365 * 24
    return pd.date_range(start, periods=periods, freq="H")

YEAR_IDX = make_year_index()

# --- PV synthetic generators (normalized per kWc) ---
def pv_annual_profile_south(normalize=True):
    # simple physical-inspired model: daily gaussian-shaped production centered at noon,
    # plus seasonal variation by day-of-year (more in summer).
    idx = YEAR_IDX
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values / 60.0
    # daylight factor: approximate daylength by sinus on DOY (longer in mid-year)
    seasonal = 0.5 + 0.5 * np.sin((doy - 80) / 365.0 * 2 * np.pi)  # peaks around june
    # solar hour angle shaped production (peak at 12h)
    daily = np.maximum(0, np.cos((hour - 12) / 12 * np.pi))
    prod = daily * seasonal
    if normalize:
        prod = prod / prod.max()
    return pd.Series(prod, index=idx)

def pv_annual_profile_east_west(normalize=True):
    idx = YEAR_IDX
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values / 60.0
    seasonal = 0.5 + 0.5 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    # bimodal: two peaks morning and afternoon (east-west)
    morning = np.maximum(0, np.cos((hour - 9) / 6 * np.pi))
    afternoon = np.maximum(0, np.cos((hour - 15) / 6 * np.pi))
    prod = (morning + afternoon) * 0.6 * seasonal
    if normalize:
        prod = prod / prod.max()
    return pd.Series(prod, index=idx)

# Precompute normalized templates
pv_south_norm = pv_annual_profile_south()
pv_ew_norm = pv_annual_profile_east_west()

# --- Load profile generators (normalized, will scale to annual energy) ---
def load_profile_template(building="Résidentiel"):
    # returns hourly pattern for a single typical day (24 values) normalized to mean=1
    h = np.arange(24)
    if building == "Résidentiel":
        # low daytime, peaks early morning and evening
        profile = 0.6 + 0.4 * np.exp(-((h - 7) ** 2) / 10) + 0.6 * np.exp(-((h - 19) ** 2) / 12)
    elif building == "Tertiaire":
        # high daytime, low nights
        profile = 0.4 + 1.2 * np.exp(-((h - 13) ** 2) / 18) * (h >= 7) * (h <= 19)
        profile = profile + 0.1  # baseline
    else:  # Industriel
        # near-constant with slight daytime increase
        profile = 1.0 + 0.2 * np.exp(-((h - 12) ** 2) / 60)
    # normalize to mean 1 so scaling to annual energy is straightforward
    profile = profile / np.mean(profile)
    return profile

def make_annual_load_from_profile(building, annual_mwh):
    # annual_mwh in MWh -> convert to kWh total
    total_kwh = annual_mwh * 1000.0
    template_daily = load_profile_template(building)
    # repeat for 365 days
    daily_profile = np.tile(template_daily, 365)
    # adjust seasonal variation small (slightly higher in winter for résidentiel)
    doy = YEAR_IDX.dayofyear.values
    seasonal = 1.0 + 0.1 * np.cos((doy - 200) / 365 * 2 * np.pi) if building == "Résidentiel" else 1.0
    annual_profile = daily_profile * seasonal
    # scale so that sum(hourly)/1000 = total_kwh (since values are in kW for each hour, sum gives kWh)
    current_sum = annual_profile.sum()
    factor = total_kwh / current_sum
    hourly_kw = annual_profile * factor  # units: kW (hourly values)
    return pd.Series(hourly_kw, index=YEAR_IDX)

# --- CSV parsers ---
def parse_csv_timevalue(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=";", header=None, encoding="utf-8", engine="python")
        # attempt to find the first row that's a date - assume data starts from row 2 (index 1) or 0
        # if header present, try parsing robustly
        # Find first column with date-like strings
        # Combine into datetime parsing tolerant with dayfirst
        # Build two-column assumption: datetime ; value
        # Drop any NaN rows
        # Accept formats dd.mm.yyyy HH:MM or yyyy-mm-dd ...
        # We'll try multiple parse attempts.
        # If file has header, detect numeric in second column possibly.
        # Flatten to two columns
        if df.shape[1] >= 2:
            col0 = df.iloc[:,0].astype(str)
            col1 = df.iloc[:,1]
            # find first row where col1 is numeric-like
            mask_numeric = pd.to_numeric(col1, errors="coerce").notna()
            # consider starting point
            if mask_numeric.any():
                first_idx = mask_numeric.idxmax()
                col0 = col0[first_idx:]
                col1 = pd.to_numeric(col1[first_idx:], errors="coerce")
                temp = pd.DataFrame({"date": col0.values, "value": col1.values})
            else:
                # fallback: try parse all rows
                temp = pd.DataFrame({"date": col0.values, "value": pd.to_numeric(col1, errors='coerce')})
        else:
            st.warning("Le CSV ne semble pas avoir au moins deux colonnes.")
            return None
        # parse date with dayfirst ; try common formats
        def try_parse(s):
            for fmt in ("%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    pass
            # try pandas
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
        # resample to hourly mean if finer resolution; if provided values are kWh per interval, user should provide kW hourly
        # We'll convert to hourly power (kW) by assuming input is instantaneous kW or average per hour.
        series = temp["value"].resample("H").mean().reindex(YEAR_IDX, method="nearest", fill_value=0)
        return series
    except Exception as e:
        st.warning(f"Erreur lecture CSV : {e}")
        return None

# --- Build PV timeseries ---
def build_pv_timeseries(pv_file, kwc, orientation, inclinaison):
    if pv_file is not None:
        series = parse_csv_timevalue(pv_file)
        if series is not None:
            # normalize to day/month only: map any year to template year by copying values by month-day-hour
            # But keep hourly series resampled to YEAR_IDX by mapping month-day-hour buckets.
            df = series.to_frame("val").reset_index()
            df["mdh"] = df["index"].dt.strftime("%m-%d %H:%M")
            # create mapping from mdh -> value by taking median over years if multiple years present
            mapping = df.groupby("mdh")["val"].median()
            # build series for YEAR_IDX using mapping; for missing keys, set 0
            idx_mdh = YEAR_IDX.strftime("%m-%d %H:%M")
            vals = [mapping.get(k, 0.0) for k in idx_mdh]
            return pd.Series(vals, index=YEAR_IDX)
        else:
            st.info("Impossible de parser le CSV PV; utilisation du modèle synthétique.")
    # synthetic model
    if orientation == "Sud":
        base = pv_south_norm.copy()
    else:
        base = pv_ew_norm.copy()
    # scale by kwc and apply simple tilt correction (approximate)
    tilt_factor = max(0.6, 1 - abs(inclinaison - 30) / 100.0)  # simple penalty if not optimal
    series = base * (kwc * tilt_factor)
    return series

# --- Build consumption timeseries ---
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

# --- Build series ---
pv_ts = build_pv_timeseries(pv_file, pv_kwc, orientation, inclinaison)
cons_ts = build_consumption_timeseries(cons_file, building_type, annual_consumption_mwh)

# --- Select date for comparison (ignore year) ---
st.sidebar.header("Comparaison")
st.sidebar.write("Choisir un jour et un mois (l'année est ignorée)")
sel_date = st.sidebar.date_input("Date de comparaison (année ignorée)", value=datetime(2001,6,21))
# Build index for that day in template year (2001)
month = sel_date.month
day = sel_date.day
day_start = datetime(2001, month, day, 0, 0)
idx_day = pd.date_range(day_start, periods=24, freq="H")

# extract day series by matching month-day-hour in YEAR_IDX mapping
def extract_day(series, month, day):
    sel = series[(series.index.month == month) & (series.index.day == day)]
    # ensure length 24 by reindexing to correct hours from YEAR_IDX matching that day
    base_idx = pd.date_range(datetime(2001, month, day, 0, 0), periods=24, freq="H")
    # map by month-day-hour string
    mdh_series = series.copy()
    mdh_series.index = mdh_series.index.strftime("%m-%d %H:%M")
    keys = base_idx.strftime("%m-%d %H:%M")
    vals = [mdh_series.get(k, 0.0) for k in keys]
    return pd.Series(vals, index=base_idx)

pv_day = extract_day(pv_ts, month, day)
cons_day = extract_day(cons_ts, month, day)

# --- Compute simple energy flows for the selected day (no BESS yet): ---
# For each hour, PV covers load up to its production; excess PV -> export; deficit -> import from grid.
pv_kwh = pv_day  # kW hourly -> kWh per hour
load_kwh = cons_day
self_consumed = np.minimum(pv_kwh, load_kwh)
exported = np.maximum(0, pv_kwh - load_kwh)
imported = np.maximum(0, load_kwh - pv_kwh)

# hourly bess placeholders (zeros for now)
bess_charge = np.zeros_like(pv_kwh)
bess_discharge = np.zeros_like(pv_kwh)

# Aggregate energies for pie charts (sum over 24 hours)
E_self = self_consumed.sum()
E_export = exported.sum()
E_import = imported.sum()
total_pv = pv_kwh.sum()
total_load = load_kwh.sum()

# --- Plotting ---
st.subheader(f"Superposition production PV vs consommation — {sel_date.strftime('%d %B')} (année synthétique)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=pv_day.index.hour, y=pv_day.values, name="Production PV (kW)", line=dict(color=COLORS["pv"]), fill='tozeroy'))
fig.add_trace(go.Scatter(x=cons_day.index.hour, y=cons_day.values, name="Consommation (kW)", line=dict(color=COLORS["load"])))
fig.update_layout(xaxis_title="Heure (h)", yaxis_title="Puissance (kW)", legend=dict(orientation="h"))

# Layout: left graph, right indicators + pies
left_col, right_col = st.columns([2.3, 1])

with left_col:
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.metric("Production PV totale (kWh)", f"{total_pv:.1f}")
    st.metric("Consommation totale (kWh)", f"{total_load:.1f}")
    st.metric("Autoconsommation (kWh)", f"{E_self:.1f}")

    st.write("### Répartition de l'énergie (24h sélectionnées)")
    pie1 = px.pie(values=[E_self, E_export], names=["Autoconsommée", "Exportée"], color_discrete_sequence=[COLORS["pv"], COLORS["grid_export"]])
    pie1.update_traces(textinfo='percent+label')
    st.plotly_chart(pie1, use_container_width=True)

    st.write("### Source d'énergie pour la consommation (24h)")
    pie2 = px.pie(values=[E_self, E_import], names=["PV (auto)", "Réseau (import)"], color_discrete_sequence=[COLORS["pv"], COLORS["grid_import"]])
    pie2.update_traces(textinfo='percent+label')
    st.plotly_chart(pie2, use_container_width=True)

st.markdown("---")
st.write("Notes :")
st.write("- Les années sont synthétiques : seules la date (jour+mois) et l'heure sont utilisées.")
st.write("- Les CSV importés sont traités en ignorant l'année : seules les combinaisons jour+mois+heure sont retrouvées.")
st.write("- À ce stade la BESS n'est pas encore modélisée : les valeurs de charge/décharge sont à zéro (placeholder).")
st.write("- Les modèles PV utilisés pour l'approximation (Sud / Est-Ouest) sont des modèles synthétiques basés sur profils typiques et un effet d'échelle par kWc.")

st.write("### Remarques techniques")
st.write("- Si vous souhaitez fournir vos propres courbes PV ou consommation, téléversez un CSV au format indiqué ci-dessus.")
st.write("- Les profils générés automatiquement sont destinés à des simulations rapides et qualitatives.")

