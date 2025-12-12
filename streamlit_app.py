import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Calculateur Rentabilité PV & Stockage", layout="wide")


def calculate_irr(investment: float, annual_cashflow: float, years: int = 20) -> float:
    """
    TRI interne (utilisé uniquement pour certains scores internes si besoin).
    Retourne un taux en %.
    """
    if investment <= 0:
        return 0.0
    if annual_cashflow <= 0:
        return -99.9

    low, high = -0.99, 10.0
    for _ in range(100):
        r = (low + high) / 2
        if r == 0:
            npv = (annual_cashflow * years) - investment
        else:
            factor = (1 - (1 + r) ** (-years)) / r
            npv = annual_cashflow * factor - investment

        if abs(npv) < 0.01:
            return r * 100
        if npv > 0:
            low = r
        else:
            high = r
    return r * 100


def simulate_energy_flow(
    pv: np.ndarray,
    load: np.ndarray,
    capacity_kwh: float,
    power_kva: float,
    step_hours: float = 1.0,
    rte: float = 0.90,
    dod: float = 1.0,
) -> dict:
    """
    Simule les flux énergétiques avec pertes symétriques (sqrt(RTE) charge et décharge).
    pv/load sont des séries d'énergie par pas (kWh par pas), pas des kW.
    power_kva est traité comme kW max (cosphi=1) et limité en énergie par pas via power_kva*step_hours.
    """
    n = len(pv)
    pv_arr = np.array(pv, dtype=float)
    load_arr = np.array(load, dtype=float)

    # PV seul
    balance = pv_arr - load_arr
    if capacity_kwh <= 0:
        exported = np.maximum(0.0, balance)
        imported = np.maximum(0.0, -balance)
        self_consumed = np.minimum(pv_arr, load_arr)
        return {
            "imported": float(np.sum(imported)),
            "exported": float(np.sum(exported)),
            "self_consumed": float(np.sum(self_consumed)),
        }

    # Batterie
    useful_capacity = max(0.0, capacity_kwh * dod)  # capacité "chimique" utile
    soc = 0.0  # kWh chimique utile
    grid_import = 0.0
    grid_export = 0.0
    self_consumed_sum = 0.0

    max_energy_per_step = max(0.0, power_kva * step_hours)  # kWh/pas
    eff_one_way = float(np.sqrt(max(0.0, min(1.0, rte))))  # sqrt(RTE)

    for i in range(n):
        p = pv_arr[i]
        l = load_arr[i]
        diff = p - l

        if diff > 0:
            # surplus -> charge
            power_available_ac = min(diff, max_energy_per_step)

            space_chemical = useful_capacity - soc
            if space_chemical <= 0:
                ac_to_battery = 0.0
            else:
                max_ac_accepted = space_chemical / eff_one_way if eff_one_way > 0 else 0.0
                ac_to_battery = min(power_available_ac, max_ac_accepted)

            energy_stored = ac_to_battery * eff_one_way
            soc += energy_stored

            to_grid = diff - ac_to_battery
            grid_export += to_grid

            # charge couverte par PV (et éventuellement batterie en charge n’aide pas ici)
            self_consumed_sum += l

        else:
            # déficit -> décharge
            needed_ac = -diff
            power_needed_limited = min(needed_ac, max_energy_per_step)

            max_ac_provided = soc * eff_one_way
            ac_from_battery = min(power_needed_limited, max_ac_provided)

            energy_extracted = (ac_from_battery / eff_one_way) if eff_one_way > 0 else 0.0
            soc -= energy_extracted
            if soc < 0:
                soc = 0.0

            to_import = needed_ac - ac_from_battery
            grid_import += to_import

            self_consumed_sum += (p + ac_from_battery)

    return {
        "imported": float(grid_import),
        "exported": float(grid_export),
        "self_consumed": float(self_consumed_sum),
    }


def load_csv_data(uploaded_file) -> tuple[np.ndarray | None, float]:
    """
    CSV format attendu (votre format) :
    - séparateur ; et décimale ,
    - 2 lignes d'en-tête à ignorer (skiprows=2),
    - valeur dans la 2e colonne (index 1).
    Renvoie (values, step_hours), où values est un tableau de kWh par pas.
    """
    if uploaded_file is None:
        return None, 1.0

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            sep=";",
            decimal=",",
            skiprows=2,
            header=None,
            engine="python",
        )
        if df.shape[1] < 2:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                sep=None,
                decimal=",",
                skiprows=2,
                header=None,
                engine="python",
            )

        col_idx = 1 if df.shape[1] > 1 else 0
        values = pd.to_numeric(df.iloc[:, col_idx], errors="coerce").dropna().values.astype(float)

        # Détection pas 15 min (~35040 points)
        if len(values) > 30000:
            step = 0.25
            max_len = 35040 + 96
            if len(values) > max_len:
                values = values[:35040]
        else:
            step = 1.0
            if len(values) > 8760:
                values = values[:8760]

        return values, step
    except Exception:
        return None, 1.0


st.title("Analyse Rentabilité PV & Stockage (Autoconsommation)")

col_left, col_right = st.columns([1, 2], gap="large")

# =========================
# Entrées
# =========================
with col_left:
    st.header("1. Entrées")

    with st.expander("Fichiers de données", expanded=True):
        file_load = st.file_uploader("Consommation (CSV)", type=["csv"])
        file_pv = st.file_uploader("Production PV (CSV)", type=["csv"])

    st.markdown("---")
    with st.expander("Paramètres Économiques", expanded=True):
        tariff_grid_buy = st.number_input("Tarif Achat Réseau (CHF/kWh)", value=0.28, format="%.3f")
        tariff_grid_sell = st.number_input("Tarif Rachat Surplus (CHF/kWh)", value=0.04, format="%.3f")

    st.markdown("---")
    with st.expander("Durée de vie", expanded=True):
        project_lifetime_years = st.number_input(
            "Durée de vie du projet (années)",
            min_value=1,
            value=20,
            step=1,
            format="%d",
        )

    st.markdown("---")
    with st.expander("Dimensionnement & Coûts", expanded=True):
        st.markdown("**Photovoltaïque**")
        pv_power_kwc = st.number_input("Puissance PV (kWc)", value=100.0, format="%.0f")
        capex_pv_kwc = st.number_input("Coût invest. PV (CHF/kWc)", value=800.0, format="%.0f")

        st.markdown("**Stockage (BESS)**")
        batt_capacity_kwh = st.number_input("Capacité Nominale Batterie (kWh)", value=50.0, format="%.0f")
        batt_power_kva = st.number_input("Puissance Onduleur (kVA)", value=50.0, format="%.0f")
        capex_batt_kwh = st.number_input("Coût invest. Stockage (CHF/kWh)", value=550.0, format="%.0f")

        st.markdown("**Paramètres Techniques Batterie**")
        batt_dod = st.slider("Profondeur de Décharge (DoD) %", min_value=50, max_value=100, value=100, step=1)
        batt_rte = st.slider("Rendement Aller-Retour (RTE) %", min_value=70, max_value=100, value=90, step=1)

        batt_dod_val = batt_dod / 100.0
        batt_rte_val = batt_rte / 100.0

    # Chargement des CSV
    data_load = None
    data_pv = None
    step_h = 1.0

    if file_load:
        data_load, step_load = load_csv_data(file_load)
    if file_pv:
        data_pv, step_pv = load_csv_data(file_pv)

    # Alignement simple
    if data_load is not None and data_pv is not None:
        if len(data_load) != len(data_pv):
            st.warning("Attention : résolutions différentes. Alignement forcé sur la série la plus courte.")
            min_len = min(len(data_load), len(data_pv))
            data_load = data_load[:min_len]
            data_pv = data_pv[:min_len]
            # On prend le pas du load si présent sinon pv (approx)
            step_h = step_load if file_load else step_pv
        else:
            step_h = step_load

# =========================
# Résultats
# =========================
with col_right:
    st.header("2. Résultats de l'analyse")

    if data_load is None or data_pv is None:
        st.info("Veuillez importer vos fichiers CSV (Consommation et Production) pour lancer l'analyse.")
    else:
        sim_mode = "15 minutes" if step_h == 0.25 else "Horaire"
        st.caption(f"Mode de simulation : {sim_mode} ({len(data_load)} points)")

        total_load = float(np.sum(data_load))
        total_pv = float(np.sum(data_pv))

        # --- A. PV seul
        st.subheader("A. Photovoltaïque Seul")

        res_pv = simulate_energy_flow(data_pv, data_load, 0.0, 0.0, step_hours=step_h)
        autoconso_pv = res_pv["self_consumed"]
        export_pv = res_pv["exported"]

        taux_autoconso_pv = (autoconso_pv / total_pv * 100) if total_pv > 0 else 0.0
        taux_autoprod_pv = (autoconso_pv / total_load * 100) if total_load > 0 else 0.0

        economie_facture_pv = autoconso_pv * tariff_grid_buy
        revenu_vente_surplus_pv = export_pv * tariff_grid_sell
        total_gain_annuel_pv = economie_facture_pv + revenu_vente_surplus_pv

        invest_pv = float(pv_power_kwc * capex_pv_kwc)
        roi_pv = (invest_pv / total_gain_annuel_pv) if total_gain_annuel_pv > 0 else 99.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Consommation", f"{total_load:,.0f} kWh")
        c2.metric("Production PV", f"{total_pv:,.0f} kWh")
        c3.metric("Investissement PV", f"{invest_pv:,.0f} CHF")

        c1, c2, c3 = st.columns(3)
        c1.metric("Énergie Autoconsommée", f"{autoconso_pv:,.0f} kWh")
        c2.metric("Taux Autoconsommation", f"{taux_autoconso_pv:.1f} %")
        c3.metric("Taux Autoproduction", f"{taux_autoprod_pv:.1f} %")

        c1, c2 = st.columns(2)
        c1.metric("Gain Total", f"{total_gain_annuel_pv:,.0f} CHF/an")
        c2.metric("Retour Invest.", f"{roi_pv:.1f} ans")

        st.markdown("---")

        # --- B. PV + batterie (scénario saisi)
        st.subheader(f"B. Impact Stockage ({batt_capacity_kwh:.0f} kWh)")

        if batt_capacity_kwh <= 0:
            st.info("Saisissez une capacité batterie > 0 pour afficher ce bloc.")
        else:
            res_batt = simulate_energy_flow(
                data_pv,
                data_load,
                float(batt_capacity_kwh),
                float(batt_power_kva),
                step_hours=step_h,
                rte=batt_rte_val,
                dod=batt_dod_val,
            )

            autoconso_batt = res_batt["self_consumed"]
            taux_autoconso_batt = (autoconso_batt / total_pv * 100) if total_pv > 0 else 0.0
            taux_autoprod_batt = (autoconso_batt / total_load * 100) if total_load > 0 else 0.0

            # Energie sauvée = énergie restituée utile au bâtiment (après pertes)
            kwh_saved = autoconso_batt - autoconso_pv

            # Gain net = achat réseau évité - manque à gagner de revente
            gain_net_batt = (kwh_saved * tariff_grid_buy) - (kwh_saved * tariff_grid_sell)

            invest_batt = float(batt_capacity_kwh * capex_batt_kwh)
            roi_batt = (invest_batt / gain_net_batt) if gain_net_batt > 0 else 99.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Tx Autoconso", f"{taux_autoconso_batt:.1f} %", delta=f"+{(taux_autoconso_batt - taux_autoconso_pv):.1f}")
            c2.metric("Tx Autoprod", f"{taux_autoprod_batt:.1f} %", delta=f"+{(taux_autoprod_batt - taux_autoprod_pv):.1f}")
            c3.metric("Énergie Sauvée", f"{kwh_saved:,.0f} kWh")

            c1, c2, c3 = st.columns(3)
            c1.metric("Investissement Batt.", f"{invest_batt:,.0f} CHF")
            c2.metric("Gain Net (Delta)", f"{gain_net_batt:,.0f} CHF/an")
            c3.metric("Retour Invest.", f"{roi_batt:.1f} ans")

        st.markdown("---")

        # --- C. Optimisation
        st.subheader("C. Optimisation Dimensionnement")

        optim_strategy = st.radio(
            "Stratégie :",
            (
                "Batterie optimale pour maximiser l'autoproduction",
                "Gain financier maximal sur la durée de vie du projet",
                "Temps de retour le plus court",
            ),
            horizontal=True,
        )

        if st.button("Lancer l'optimisation"):
            with st.spinner("Calcul en cours..."):
                best_score = -1e30
                best_cap = 0.0
                best = {}

                # Plage test (heuristique)
                avg_daily = total_load / 365.0 if total_load > 0 else 0.0
                max_cap = max(10.0, avg_daily * 2.0)
                caps_to_test = np.linspace(5.0, max_cap, 30)

                for cap in caps_to_test:
                    r = simulate_energy_flow(
                        data_pv,
                        data_load,
                        float(cap),
                        float(batt_power_kva),
                        step_hours=step_h,
                        rte=batt_rte_val,
                        dod=batt_dod_val,
                    )

                    autoprod_pct = (r["self_consumed"] / total_load * 100) if total_load > 0 else 0.0
                    kwh_saved_opt = r["self_consumed"] - autoconso_pv
                    gain_net_opt = (kwh_saved_opt * tariff_grid_buy) - (kwh_saved_opt * tariff_grid_sell)
                    inv_opt = float(cap * capex_batt_kwh)
                    payback_opt = (inv_opt / gain_net_opt) if gain_net_opt > 0 else 1e30

                    cash_net_lifetime = (gain_net_opt * project_lifetime_years) - inv_opt if gain_net_opt > 0 else -1e30

                    if optim_strategy == "Batterie optimale pour maximiser l'autoproduction":
                        # Option: exiger gain>0 pour éviter batteries non rentables
                        score = autoprod_pct if gain_net_opt > 0 else -1e30

                    elif optim_strategy == "Gain financier maximal sur la durée de vie du projet":
                        score = cash_net_lifetime

                    else:  # Temps de retour le plus court
                        score = (-payback_opt) if gain_net_opt > 0 else -1e30

                    if score > best_score:
                        best_score = score
                        best_cap = float(cap)
                        best = {
                            "autoprod_pct": autoprod_pct,
                            "gain_net": gain_net_opt,
                            "invest": inv_opt,
                            "payback": payback_opt,
                            "cash_net_lifetime": cash_net_lifetime,
                        }

                if best and best["gain_net"] > 0:
                    st.success(f"Batterie optimale : {best_cap:.0f} kWh")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Capacité Optimale", f"{best_cap:.0f} kWh")
                    c2.metric("Temps de Retour", f"{best['payback']:.1f} ans")
                    c3.metric("Autoproduction", f"{best['autoprod_pct']:.1f} %")

                    st.caption(
                        f"Investissement : {best['invest']:,.0f} CHF | "
                        f"Gain net annuel : {best['gain_net']:,.0f} CHF/an | "
                        f"Cash net sur {project_lifetime_years} ans : {best['cash_net_lifetime']:,.0f} CHF"
                    )
                else:
                    st.warning("Aucune batterie rentable trouvée (gain net annuel ≤ 0).")