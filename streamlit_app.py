import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Calculateur Rentabilité PV & Stockage", layout="wide")


# --- Fonctions de Simulation ---

def calculate_irr(investment, annual_cashflow, years=20):
    """Calcule le TRI sur une durée donnée."""
    if investment <= 0: return 0.0
    if annual_cashflow <= 0: return -99.9

    low, high = -0.99, 10.0
    for _ in range(100):
        r = (low + high) / 2
        if r == 0:
            npv = (annual_cashflow * years) - investment
        else:
            factor = (1 - (1 + r) ** (-years)) / r
            npv = annual_cashflow * factor - investment
        if abs(npv) < 0.01: return r * 100
        if npv > 0:
            low = r
        else:
            high = r
    return r * 100


def simulate_energy_flow(pv, load, capacity_kwh, power_kva, step_hours=1.0, rte=0.90, dod=1.0):
    """
    Simule les flux énergétiques.
    Retourne les indicateurs globaux ET les séries temporelles de charge/décharge.
    """
    n = len(pv)
    pv_arr = np.array(pv, dtype=float)
    load_arr = np.array(load, dtype=float)

    charge_series = np.zeros(n)
    discharge_series = np.zeros(n)

    balance = pv_arr - load_arr

    if capacity_kwh <= 0:
        exported = np.maximum(0.0, balance)
        imported = np.maximum(0.0, -balance)
        self_consumed = np.minimum(pv_arr, load_arr)
        return {
            "imported": float(np.sum(imported)),
            "exported": float(np.sum(exported)),
            "self_consumed": float(np.sum(self_consumed)),
            "daily_charge": np.zeros(365),
            "daily_discharge": np.zeros(365),
            "total_discharge": 0.0
        }

    useful_capacity = capacity_kwh * dod
    soc = 0.0
    grid_import = 0.0
    grid_export = 0.0
    self_consumed_sum = 0.0

    max_energy_per_step = power_kva * step_hours
    eff_one_way = np.sqrt(rte)

    for i in range(n):
        p = pv_arr[i]
        l = load_arr[i]
        diff = p - l

        if diff > 0:
            space_chemical = useful_capacity - soc
            max_ac_accepted = space_chemical / eff_one_way
            ac_to_battery = min(diff, max_energy_per_step, max_ac_accepted)

            charge_series[i] = ac_to_battery
            soc += ac_to_battery * eff_one_way
            grid_export += (diff - ac_to_battery)
            self_consumed_sum += l
        else:
            needed_ac = -diff
            max_ac_provided = soc * eff_one_way
            ac_from_battery = min(needed_ac, max_energy_per_step, max_ac_provided)

            discharge_series[i] = ac_from_battery
            soc -= (ac_from_battery / eff_one_way)
            if soc < 0: soc = 0.0

            grid_import += (needed_ac - ac_from_battery)
            self_consumed_sum += (p + ac_from_battery)

    points_per_day = int(n / 365)
    daily_charge = charge_series[:365 * points_per_day].reshape(365, points_per_day).sum(axis=1)
    daily_discharge = discharge_series[:365 * points_per_day].reshape(365, points_per_day).sum(axis=1)

    return {
        "imported": float(grid_import),
        "exported": float(grid_export),
        "self_consumed": float(self_consumed_sum),
        "daily_charge": daily_charge,
        "daily_discharge": daily_discharge,
        "total_discharge": float(np.sum(discharge_series))
    }


def load_csv_data(uploaded_file):
    if uploaded_file is None: return None, 1.0
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', skiprows=2, header=None, engine='python')
        col_idx = 1 if df.shape[1] > 1 else 0
        values = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values.astype(float)
        step = 0.25 if len(values) > 30000 else 1.0
        target_len = 35040 if step == 0.25 else 8760
        if len(values) > target_len:
            values = values[:target_len]
        elif len(values) < target_len:
            values = np.pad(values, (0, target_len - len(values)), 'constant')
        return values, step
    except Exception:
        return None, 1.0


# --- Interface ---

st.title("Analyse Rentabilité PV & Stockage (Autoconsommation)")

col_left, col_right = st.columns([1, 2], gap="large")

# ================= GAUCHE : ENTRÉES =================
with col_left:
    st.header("1. Entrées")

    with st.expander("Fichiers de données", expanded=True):
        file_load = st.file_uploader("Consommation (CSV)", type=["csv"])
        file_pv = st.file_uploader("Production PV (CSV)", type=["csv"])

    st.markdown("---")
    with st.expander("Paramètres Économiques", expanded=True):
        tariff_grid_buy = st.number_input("Tarif Achat Réseau (CHF/kWh)", value=0.28, format="%.3f")
        tariff_grid_sell = st.number_input("Tarif Rachat Surplus (CHF/kWh)", value=0.04, format="%.3f")
        project_lifetime_years = st.number_input("Durée de vie projet (ans)", min_value=1, value=20, step=1)
        opex_pv_val = st.number_input("OPEX PV (CHF / kWc / an)", value=15.0, format="%.1f")
        opex_batt_val = st.number_input("OPEX Batterie (CHF / kWh / an)", value=10.0, format="%.1f")

    st.markdown("---")
    with st.expander("Dimensionnement & Coûts", expanded=True):
        st.markdown("**Photovoltaïque**")
        pv_power_kwc = st.number_input("Puissance PV (kWc)", value=100.0, format="%.0f")
        capex_pv_kwc = st.number_input("Coût invest. PV (CHF/kWc)", value=800.0, format="%.0f")

        st.markdown("**Stockage (BESS)**")
        batt_capacity_kwh = st.number_input("Capacité Nominale Batterie (kWh)", value=50.0, format="%.0f")
        batt_power_kva = st.number_input("Puissance Onduleur (kVA)", value=50.0, format="%.0f")
        capex_batt_kwh = st.number_input("Coût invest. Stockage (CHF/kWh)", value=550.0, format="%.0f")

        st.markdown("**Technique Batterie**")
        batt_dod = st.slider("DoD %", 50, 100, 100)
        batt_rte = st.slider("Rendement Aller-Retour (RTE) %", 70, 100, 90)

        batt_dod_val, batt_rte_val = batt_dod / 100.0, batt_rte / 100.0

    data_load, data_pv = None, None
    step_h = 1.0
    if file_load: data_load, step_load = load_csv_data(file_load)
    if file_pv: data_pv, step_pv = load_csv_data(file_pv)
    if data_load is not None and data_pv is not None:
        step_h = step_load

# ================= DROITE : RÉSULTATS =================
with col_right:
    st.header("2. Résultats de l'analyse")

    if data_load is None or data_pv is None:
        st.info("👋 Veuillez importer vos fichiers CSV.")
    else:
        total_load = np.sum(data_load)
        total_pv = np.sum(data_pv)

        # --- A. PV SEUL ---
        st.subheader("A. Photovoltaïque Seul")
        res_pv = simulate_energy_flow(data_pv, data_load, 0, 0, step_hours=step_h)
        autoconso_pv = res_pv['self_consumed']

        invest_pv = pv_power_kwc * capex_pv_kwc
        opex_pv_annuel = pv_power_kwc * opex_pv_val

        c1, c2, c3 = st.columns(3)
        c1.metric("Consommation", f"{total_load:,.0f} kWh")
        c2.metric("Production PV", f"{total_pv:,.0f} kWh")
        c3.metric("Invest. PV", f"{invest_pv:,.0f} CHF")

        c1, c2, c3 = st.columns(3)
        c1.metric("Autoconsommation", f"{(autoconso_pv / total_pv * 100):.1f} %")
        c2.metric("Autoproduction", f"{(autoconso_pv / total_load * 100):.1f} %")

        gain_pv_brut = (autoconso_pv * tariff_grid_buy) + (res_pv['exported'] * tariff_grid_sell)
        gain_pv_net = gain_pv_brut - opex_pv_annuel
        roi_pv = invest_pv / gain_pv_net if gain_pv_net > 0 else 99

        c3.metric("Gain Annuel (Net)", f"{gain_pv_net:,.0f} CHF")

        c1, c2 = st.columns(2)
        c1.metric("OPEX PV", f"{opex_pv_annuel:,.0f} CHF/an")
        c2.metric("Retour Invest.", f"{roi_pv:.1f} ans")

        st.markdown("---")

        # --- B. IMPACT STOCKAGE ---
        st.subheader(f"B. Impact Stockage ({batt_capacity_kwh:.0f} kWh)")

        if batt_capacity_kwh > 0:
            res_b = simulate_energy_flow(data_pv, data_load, batt_capacity_kwh, batt_power_kva,
                                         step_hours=step_h, rte=batt_rte_val, dod=batt_dod_val)

            kwh_sauves = res_b['self_consumed'] - autoconso_pv

            invest_b = batt_capacity_kwh * capex_batt_kwh
            opex_b_annuel = batt_capacity_kwh * opex_batt_val

            gain_b_brut = (kwh_sauves * tariff_grid_buy) - (kwh_sauves * tariff_grid_sell)
            gain_b_net = gain_b_brut - opex_b_annuel

            roi_b = invest_b / gain_b_net if gain_b_net > 0 else 99
            cash_net_b = (gain_b_net * project_lifetime_years) - invest_b

            c1, c2, c3 = st.columns(3)
            c1.metric("Nouveau Taux Autoconso", f"{(res_b['self_consumed'] / total_pv * 100):.1f} %",
                      delta=f"+{(res_b['self_consumed'] / total_pv * 100 - autoconso_pv / total_pv * 100):.1f} pts")
            c2.metric("Nouveau Taux Autoprod", f"{(res_b['self_consumed'] / total_load * 100):.1f} %",
                      delta=f"+{(kwh_sauves / total_load * 100):.1f} pts")
            c3.metric("Énergie Sauvée", f"{kwh_sauves:,.0f} kWh")

            cf1, cf2, cf3, cf4 = st.columns(4)
            cf1.metric("Gain Net Batt.", f"{gain_b_net:,.0f} CHF/an")
            cf2.metric("Investissement Batt.", f"{invest_b:,.0f} CHF")
            cf3.metric("Retour Invest.", f"{roi_b:.1f} ans")
            cf4.metric("Cash Net Projet", f"{cash_net_b:,.0f} CHF")

            # KPI Cycles
            useful_cap = batt_capacity_kwh * batt_dod_val
            nb_cycles_complets = res_b['total_discharge'] / useful_cap if useful_cap > 0 else 0
            nb_jours_actifs = np.count_nonzero(res_b['daily_charge'] > 0.1)
            avg_fill = (res_b['daily_discharge'].mean() / useful_cap * 100) if useful_cap > 0 else 0

            st.write("### Sollicitation de la batterie")
            col_cyc1, col_cyc2, col_cyc3 = st.columns(3)
            col_cyc1.metric("Nombre de cycles / an", f"{nb_jours_actifs}")
            col_cyc2.metric("Nombre de cycles complets / an", f"{nb_cycles_complets:.0f}")
            col_cyc3.metric("Remplissage journalier moyen", f"{avg_fill:.1f} %")

            # Graphique
            days = np.arange(1, 366)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=days, y=res_b['daily_charge'], name="Charge (Solaire vers Batterie)", marker_color='#2ECC71'))
            fig.add_trace(go.Bar(x=days, y=-res_b['daily_discharge'], name="Décharge (Batterie vers Bâtiment)",
                                 marker_color='#E74C3C'))
            fig.update_layout(title="Flux journaliers cumulés (kWh/jour)", barmode='relative', height=350,
                              margin=dict(l=20, r=20, t=40, b=20),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Saisissez une capacité.")

        st.markdown("---")

        # --- C. OPTIMISATION ---
        st.subheader("C. Optimisation Dimensionnement")
        optim_strat = st.radio("Stratégie :", ("Batterie optimale pour maximiser l'autoproduction",
                                               "Gain financier maximal sur la durée de vie du projet",
                                               "Temps de retour le plus court"), horizontal=True)

        if st.button("Lancer l'optimisation"):
            with st.spinner("Optimisation..."):
                best_score, best_cap, best_m = -1e30, 0, {}
                caps = np.linspace(5, (total_load / 365) * 2.0, 30)
                for c in caps:
                    r = simulate_energy_flow(data_pv, data_load, c, batt_power_kva, step_hours=step_h, rte=batt_rte_val,
                                             dod=batt_dod_val)
                    k_s = r['self_consumed'] - autoconso_pv
                    inv_opt = c * capex_batt_kwh
                    opex_opt = c * opex_batt_val
                    g_n = (k_s * tariff_grid_buy) - (k_s * tariff_grid_sell) - opex_opt
                    autop_load = (r['self_consumed'] / total_load) * 100
                    payback = inv_opt / g_n if g_n > 0 else 1e30
                    cash_life = (g_n * project_lifetime_years) - inv_opt if g_n > 0 else -1e30

                    if optim_strat == "Batterie optimale pour maximiser l'autoproduction":
                        score = autop_load if g_n > 0 else -1e30
                    elif optim_strat == "Gain financier maximal sur la durée de vie du projet":
                        score = cash_life
                    else:
                        score = -payback if g_n > 0 else -1e30

                    if score > best_score:
                        best_score, best_cap = score, c
                        best_m = {"payback": payback, "autop_load": autop_load, "gain": g_n, "inv": inv_opt,
                                  "cash_life": cash_life, "res": r}

                if best_m.get("gain", -1) > 0:
                    st.success(f"Batterie optimale : {best_cap:.0f} kWh")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Capacité Optimale", f"{best_cap:.0f} kWh")
                    col2.metric("Temps de Retour", f"{best_m['payback']:.1f} ans")
                    col3.metric("Autoproduction", f"{best_m['autop_load']:.1f} %")
                    st.caption(
                        f"Investissement : {best_m['inv']:,.0f} CHF | Gain net annuel : {best_m['gain']:,.0f} CHF/an | Cash net projet : {best_m['cash_life']:,.0f} CHF")

                    # Graph et KPI pour l'optimisation
                    opt_res = best_m['res']
                    u_cap_opt = best_cap * batt_dod_val
                    nb_c_opt = opt_res['total_discharge'] / u_cap_opt if u_cap_opt > 0 else 0
                    nb_j_opt = np.count_nonzero(opt_res['daily_charge'] > 0.1)
                    avg_f_opt = (opt_res['daily_discharge'].mean() / u_cap_opt * 100) if u_cap_opt > 0 else 0

                    st.write("### Sollicitation de la batterie optimale")
                    cx1, cx2, cx3 = st.columns(3)
                    cx1.metric("Nombre de cycles / an", f"{nb_j_opt}")
                    cx2.metric("Nombre de cycles complets / an", f"{nb_c_opt:.0f}")
                    cx3.metric("Remplissage journalier moyen", f"{avg_f_opt:.1f} %")

                    fig_opt = go.Figure()
                    days = np.arange(1, 366)
                    fig_opt.add_trace(go.Bar(x=days, y=opt_res['daily_charge'], name="Charge (Solaire vers Batterie)",
                                             marker_color='#2ECC71'))
                    fig_opt.add_trace(
                        go.Bar(x=days, y=-opt_res['daily_discharge'], name="Décharge (Batterie vers Bâtiment)",
                               marker_color='#E74C3C'))
                    fig_opt.update_layout(title=f"Flux journaliers - Config optimale ({best_cap:.0f} kWh)",
                                          barmode='relative', height=350, margin=dict(l=20, r=20, t=40, b=20),
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_opt, use_container_width=True)
                else:
                    st.warning("Aucune rentabilité trouvée.")