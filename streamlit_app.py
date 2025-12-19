import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Calculateur Rentabilité PV & BESS", layout="wide")

# --- Fonctions de Simulation ---

def calculate_irr(investment, annual_cashflow, years=20):
    if investment <= 0: return 0.0
    if annual_cashflow <= 0: return -99.9
    low, high = -0.99, 10.0
    for _ in range(100):
        r = (low + high) / 2
        if r == 0: npv = (annual_cashflow * years) - investment
        else:
            factor = (1 - (1 + r) ** (-years)) / r
            npv = annual_cashflow * factor - investment
        if abs(npv) < 0.01: return r * 100
        if npv > 0: low = r
        else: high = r
    return r * 100

def simulate_energy_flow(pv, load, capacity_kwh, power_kva, step_hours=1.0, rte=0.90, dod=1.0):
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
    daily_charge = charge_series[:365*points_per_day].reshape(365, points_per_day).sum(axis=1)
    daily_discharge = discharge_series[:365*points_per_day].reshape(365, points_per_day).sum(axis=1)

    return {
        "imported": float(grid_import), "exported": float(grid_export), "self_consumed": float(self_consumed_sum),
        "daily_charge": daily_charge, "daily_discharge": daily_discharge, "total_discharge": float(np.sum(discharge_series))
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
        if len(values) > target_len: values = values[:target_len]
        elif len(values) < target_len: values = np.pad(values, (0, target_len - len(values)), 'constant')
        return values, step
    except Exception: return None, 1.0

# --- Interface ---

st.title("Calculateur de Rentabilité PV & BESS")

col_left, col_right = st.columns([1, 2], gap="large")

# ================= GAUCHE : ENTRÉES =================
with col_left:
    st.header("1. Entrées")
    
    is_contracting = st.radio("Modèle d'investissement :", ["Investissement Propre", "Tiers-Investisseur (Contracting)"]) == "Tiers-Investisseur (Contracting)"
    
    with st.expander("Fichiers de données", expanded=True):
        file_load = st.file_uploader("Consommation (CSV)", type=["csv"])
        file_pv = st.file_uploader("Production PV (CSV)", type=["csv"])
    
    st.markdown("---")
    with st.expander("Paramètres Économiques", expanded=True):
        tariff_grid_buy = st.number_input("Tarif Achat Réseau (CHF/kWh)", value=0.28, format="%.3f")
        
        # Grisé si contracting car le surplus ne revient pas au client
        tariff_grid_sell = st.number_input("Tarif Rachat Surplus (CHF/kWh)", value=0.04 if not is_contracting else 0.0, format="%.3f", disabled=is_contracting)
        
        # Actif uniquement en contracting
        tariff_pv_buy = st.number_input("Tarif Achat Solaire PV (CHF/kWh)", value=0.20, format="%.3f", disabled=not is_contracting)
        
        project_lifetime_years = st.number_input("Durée de vie projet (ans)", min_value=1, value=20, step=1)
        
        # Grisé si contracting car pas à la charge du client
        opex_pv_val = st.number_input("OPEX PV (CHF / kWc / an)", value=15.0 if not is_contracting else 0.0, format="%.1f", disabled=is_contracting)
        opex_batt_val = st.number_input("OPEX Batterie (CHF / kWh / an)", value=10.0, format="%.1f")
    
    st.markdown("---")
    with st.expander("Dimensionnement & Coûts", expanded=True):
        st.markdown("**Photovoltaïque**")
        pv_power_kwc = st.number_input("Puissance PV (kWc)", value=100.0, format="%.0f")
        
        # Grisé si contracting
        capex_pv_kwc = st.number_input("Coût invest. PV (CHF/kWc)", value=800.0 if not is_contracting else 0.0, format="%.0f", disabled=is_contracting)
        
        st.markdown("**Stockage (BESS)**")
        batt_capacity_kwh = st.number_input("Capacité Nominale Batterie (kWh)", value=50.0, format="%.0f")
        batt_power_kva = st.number_input("Puissance Onduleur (kVA)", value=50.0, format="%.0f")
        capex_batt_kwh = st.number_input("Coût invest. Stockage (CHF/kWh)", value=550.0, format="%.0f")

        st.markdown("**Technique Batterie**")
        batt_dod = st.slider("DoD %", 50, 100, 100)
        batt_rte = st.slider("Rendement (RTE) %", 70, 100, 90)
        batt_dod_val, batt_rte_val = batt_dod/100.0, batt_rte/100.0

    data_load, data_pv = None, None
    step_h = 1.0
    if file_load: data_load, step_load = load_csv_data(file_load)
    if file_pv: data_pv, step_pv = load_csv_data(file_pv)
    if data_load is not None and data_pv is not None: step_h = step_load

# ================= DROITE : RÉSULTATS =================
with col_right:
    st.header("2. Résultats de l'analyse")
    
    if data_load is None or data_pv is None:
        st.info("👋 Veuillez importer vos fichiers CSV.")
    else:
        total_load, total_pv = np.sum(data_load), np.sum(data_pv)
        
        # --- A. PV SEUL ---
        st.subheader("A. Photovoltaïque Seul")
        res_pv = simulate_energy_flow(data_pv, data_load, 0, 0, step_hours=step_h)
        ac_pv = res_pv['self_consumed']
        inv_pv = pv_power_kwc * capex_pv_kwc
        opex_pv = pv_power_kwc * opex_pv_val
        
        if not is_contracting:
            gain_pv = (ac_pv * tariff_grid_buy) + (res_pv['exported'] * tariff_grid_sell) - opex_pv
        else:
            # En contracting, gain = (tarif reseau - tarif solaire) x kWh autoconsommé
            gain_pv = ac_pv * (tariff_grid_buy - tariff_pv_buy)
            
        roi_pv = inv_pv / gain_pv if gain_pv > 0 else 99
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Consommation", f"{total_load:,.0f} kWh")
        c2.metric("Production PV", f"{total_pv:,.0f} kWh")
        c3.metric("Invest. PV", f"{inv_pv:,.0f} CHF")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Autoconsommation", f"{(ac_pv/total_pv*100):.1f} %")
        c2.metric("Autoproduction", f"{(ac_pv/total_load*100):.1f} %")
        c3.metric("Gain Annuel Net", f"{gain_pv:,.0f} CHF")
        
        if not is_contracting and inv_pv > 0: st.metric("Retour Invest. PV", f"{roi_pv:.1f} ans")

        st.markdown("---")
        
        # --- B. IMPACT STOCKAGE ---
        st.subheader(f"B. Impact Stockage ({batt_capacity_kwh:.0f} kWh)")
        if batt_capacity_kwh > 0:
            res_b = simulate_energy_flow(data_pv, data_load, batt_capacity_kwh, batt_power_kva, step_hours=step_h, rte=batt_rte_val, dod=batt_dod_val)
            kwh_s = res_b['self_consumed'] - ac_pv
            inv_b = batt_capacity_kwh * capex_batt_kwh
            opex_b = batt_capacity_kwh * opex_batt_val
            
            if not is_contracting:
                gain_b = (kwh_s * tariff_grid_buy) - (kwh_s * tariff_grid_sell) - opex_b
            else:
                gain_b = (kwh_s * (tariff_grid_buy - tariff_pv_buy)) - opex_b
                
            roi_b = inv_b / gain_b if gain_b > 0 else 99
            cash_b = (gain_b * project_lifetime_years) - inv_b
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Nouveau Tx Autoconso", f"{(res_b['self_consumed']/total_pv*100):.1f} %")
            c2.metric("Nouveau Tx Autoprod", f"{(res_b['self_consumed']/total_load*100):.1f} %")
            c3.metric("Énergie Sauvée", f"{kwh_s:,.0f} kWh")
            
            cf1, cf2, cf3, cf4 = st.columns(4)
            cf1.metric("Gain Net Batt.", f"{gain_b:,.0f} CHF/an")
            cf2.metric("Invest. Batt.", f"{inv_b:,.0f} CHF")
            cf3.metric("Retour Invest.", f"{roi_b:.1f} ans")
            cf4.metric("Cash Net Projet", f"{cash_b:,.0f} CHF")

            u_cap = batt_capacity_kwh * batt_dod_val
            nb_cc = res_b['total_discharge'] / u_cap if u_cap > 0 else 0
            avg_f = (res_b['daily_discharge'].mean() / u_cap * 100) if u_cap > 0 else 0
            st.write("### Sollicitation de la batterie")
            cy1, cy2 = st.columns(2); cy1.metric("Cycles complets / an", f"{nb_cc:.0f}"); cy2.metric("Remplissage moy / jour", f"{avg_f:.1f} %")
            
            days = np.arange(1, 366); fig = go.Figure()
            fig.add_trace(go.Bar(x=days, y=res_b['daily_charge'], name="Charge (Solaire vers Batterie)", marker_color='#2ECC71'))
            fig.add_trace(go.Bar(x=days, y=-res_b['daily_discharge'], name="Décharge (Batterie vers Bâtiment)", marker_color='#E74C3C'))
            fig.update_layout(title="Flux journaliers cumulés (kWh/jour)", barmode='relative', height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Saisissez une capacité.")

        st.markdown("---")
        
        # --- C. OPTIMISATION ---
        st.subheader("C. Optimisation Dimensionnement")
        optim_strat = st.radio("Stratégie :", ["Batterie optimale pour maximiser l'autoproduction", "Gain financier maximal", "Temps de retour le plus court"], horizontal=True)
        
        if st.button("Lancer l'optimisation"):
            with st.spinner("Calcul..."):
                best_score, best_cap, best_m = -1e30, 0, {}
                caps = np.linspace(5, (total_load/365)*2.0, 30)
                for c in caps:
                    r = simulate_energy_flow(data_pv, data_load, c, batt_power_kva, step_hours=step_h, rte=batt_rte_val, dod=batt_dod_val)
                    ks = r['self_consumed'] - ac_pv
                    inv_o = c * capex_batt_kwh
                    opex_o = c * opex_batt_val
                    if not is_contracting: go = (ks * tariff_grid_buy) - (ks * tariff_grid_sell) - opex_o
                    else: go = (ks * (tariff_grid_buy - tariff_pv_buy)) - opex_o
                    ap = (r['self_consumed'] / total_load) * 100
                    ap_pv = (r['self_consumed'] / total_pv) * 100
                    pb = inv_o / go if go > 0 else 1e30
                    cl = (go * project_lifetime_years) - inv_o
                    
                    if optim_strat == "Batterie optimale pour maximiser l'autoproduction": score = ap if go > 0 else -1e30
                    elif optim_strat == "Gain financier maximal": score = cl
                    else: score = -pb if go > 0 else -1e30
                    
                    if score > best_score:
                        best_score, best_cap = score, c
                        best_m = {"pb": pb, "ap_load": ap, "ap_pv": ap_pv, "g": go, "inv": inv_o, "cl": cl, "res": r}
                
                if best_m.get("g", -1) > 0:
                    st.success(f"Batterie optimale : {best_cap:.0f} kWh")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Capacité", f"{best_cap:.0f} kWh"); col2.metric("Retour", f"{best_m['pb']:.1f} ans"); col3.metric("Autoproduction", f"{best_m['ap_load']:.1f} %"); col4.metric("Autoconsommation", f"{best_m['ap_pv']:.1f} %")
                    st.caption(f"Investissement : {best_m['inv']:,.0f} CHF | Gain net annuel : {best_m['g']:,.0f} CHF | Cash net projet : {best_m['cl']:,.0f} CHF")
                    
                    opt_res = best_m['res']; u_c = best_cap * batt_dod_val; n_c = opt_res['total_discharge'] / u_c if u_c > 0 else 0; a_f = (opt_res['daily_discharge'].mean() / u_c * 100) if u_c > 0 else 0
                    st.write("### Sollicitation batterie optimale"); cx1, cx2 = st.columns(2); cx1.metric("Cycles complets / an", f"{n_c:.0f}"); cx2.metric("Remplissage moyen / jour", f"{a_f:.1f} %")
                    days = np.arange(1, 366); fig_o = go.Figure(); fig_o.add_trace(go.Bar(x=days, y=opt_res['daily_charge'], name="Charge", marker_color='#2ECC71')); fig_o.add_trace(go.Bar(x=days, y=-opt_res['daily_discharge'], name="Décharge", marker_color='#E74C3C'))
                    fig_o.update_layout(title=f"Flux optimaux ({best_cap:.0f} kWh)", barmode='relative', height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_o, use_container_width=True)
                else: st.warning("Aucune rentabilité trouvée.")