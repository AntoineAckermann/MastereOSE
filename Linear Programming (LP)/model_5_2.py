import pyomo.environ as pyo
import pandas as pd
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR) # éviter les messages d'erreurs


periods = {"0h-6h": [15000, 6], # [demande, durée de la période (pour le calcul des MWh dans l'objectif)]
          "6h-9h": [30000, 3],
          "9h-15h": [25000, 6],
          "15h-18h": [40000, 3],
          "18h-24h": [27000, 6]}

data_periods = pd.DataFrame.from_dict(periods, orient='index', columns=["Demand", "Hours"])


data_thermal = { "A": ["thermal", 12, 850, 2000, 2000, 1000, 2.0], # Type, N, Pmin, Pmax, Cstart, Cbase, Cmwh
         "B": ["thermal", 10, 1250, 1750, 1000, 2600, 1.3],
         "C": ["thermal", 5, 1500, 4000, 500, 3000, 3] }

data_hydro = { "H1": ["hydro", 1, 0, 900, 1500, 90, 0, 0.31],
               "H2": ["hydro", 1, 0, 1400, 1200, 150, 0, 0.47] } # N, Pmin, Pmax, Cstart, Cbase, Cmwh, water_consumption



data_thermal = pd.DataFrame.from_dict(data_thermal, orient='index', columns=["Type", "N", "Pmin", "Pmax", "Cstart", "Cbase", "Cmwh"])
data_hydro = pd.DataFrame.from_dict(data_hydro, orient='index', columns=["Type", "N", "Pmin", "Pmax", "Cstart", "Cbase", "Cmwh", "water_consumption"])


def build_model(df_thermal, df_hydro, df_periods, opt, cyc=True):
    if cyc:
            non_cyc_model = build_model(df_thermal, df_hydro, df_periods, opt, cyc=False) # il faut d'abord faire tourner le modèle avec planification non-cyclique pour obtenir les dernières valeurs pour les intégrer dans le modèle cyclique
            opt.solve(non_cyc_model)
            last_t = non_cyc_model.T.last()
            n_last = {p: pyo.value(b.n[p, last_t]) for b in non_cyc_model.Blocks for p in b.Plants}
    # 0. MODELE GLOBAL

    m = pyo.ConcreteModel("Daily planning")

    m.T = pyo.Set(initialize=df_periods.index.tolist(), ordered=True)

    for col in df_periods.columns:  # pour chaque pas de temps : la demande et durée associée à chaque période
        setattr(m, col, pyo.Param(m.T, initialize=df_periods[col].to_dict()))

    m.total_demand = pyo.Var(m.T, within=pyo.NonNegativeReals)

    # 1. CENTRALES THERMIQUES

    m.Thermal = pyo.Block()
    m.Thermal.Plants = pyo.Set(initialize=df_thermal.index.tolist(), ordered=True)

    for col in df_thermal.columns:
        setattr(m.Thermal, col, pyo.Param(m.Thermal.Plants, initialize=df_thermal[col].to_dict()))

    m.Thermal.prod = pyo.Var(m.Thermal.Plants, m.T, within=pyo.NonNegativeReals)
    m.Thermal.n = pyo.Var(m.Thermal.Plants, m.T, within=pyo.NonNegativeIntegers)
    m.Thermal.n_start_up = pyo.Var(m.Thermal.Plants, m.T, within=pyo.NonNegativeIntegers)


    # 2. CENTRALES HYDRAULIQUES

    elec_consumption = 3000 # MWh/m

    m.Hydro = pyo.Block()

    m.Hydro.Plants = pyo.Set(initialize=df_hydro.index.get_level_values(0).unique().tolist(), ordered=True)

    m.Hydro.Pmax = pyo.Param(m.Hydro.Plants, initialize=df_hydro["Pmax"].to_dict(), mutable=False)
    m.Hydro.Pmin = pyo.Param(m.Hydro.Plants, initialize=df_hydro["Pmin"].to_dict(), mutable=False)
    m.Hydro.N = pyo.Param(m.Hydro.Plants, initialize=df_hydro["N"].to_dict(), mutable=False)
    m.Hydro.Cbase = pyo.Param(m.Hydro.Plants, initialize=df_hydro["Cbase"].to_dict(), mutable=False)
    m.Hydro.Cstart = pyo.Param(m.Hydro.Plants, initialize=df_hydro["Cstart"].to_dict(), mutable=False)
    m.Hydro.water_consumption = pyo.Param(m.Hydro.Plants, initialize=df_hydro["water_consumption"].to_dict(),
                                          mutable=False)
    m.Hydro.elec_consumption = pyo.Param(initialize=elec_consumption, mutable=False)


    m.Hydro.prod = pyo.Var(m.Hydro.Plants, m.T, within=pyo.NonNegativeReals)
    m.Hydro.n = pyo.Var(m.Hydro.Plants, m.T, within=pyo.NonNegativeIntegers)
    m.Hydro.n_start_up = pyo.Var(m.Hydro.Plants, m.T,
                                 within=pyo.NonNegativeIntegers)  # nombres de centrales nouvellement allumées à chaque pas de temps

    m.Hydro.flow_in = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.Hydro.flow_out = pyo.Var(m.Hydro.Plants, m.T, within=pyo.NonNegativeReals)

    m.Blocks = [m.Thermal, m.Hydro]

    # 3. CONTRAINTES

    # 3.0 CONTRAINTES GLOBALES

    def define_demand(m, t):
        return m.total_demand[t] == m.Demand[t] + m.Hydro.flow_in[t] * m.Hydro.elec_consumption / m.Hours[t] # la consommation pour remonter le niveau d'eau se rajoute à la demande totale

    def meet_demand(m, t):
        prod_thermal = sum(m.Thermal.prod[p, t] for p in m.Thermal.Plants)
        prod_hydro = sum(m.Hydro.prod[p, t] for p in m.Hydro.Plants)

        return prod_thermal + prod_hydro >= m.total_demand[t]

    def power_reserve(m, t):
        margin = 0.15

        margin_thermal = sum(m.Thermal.Pmax[p] * m.Thermal.n[p, t] - m.Thermal.prod[p, t] for p in m.Thermal.Plants)
        margin_hydro = sum(
            m.Hydro.Pmax[p] * m.Hydro.n[p, t] - m.Hydro.prod[p, t] for p in
            m.Hydro.Plants)

        return margin_thermal + margin_hydro >= margin * m.total_demand[t]  # la somme des centrales qui produisent déjà doit pouvoir encaisser une hausse de 15% de la demande


    m.meet_demand = pyo.Constraint(m.T, rule=meet_demand)
    m.power_reserve = pyo.Constraint(m.T, rule=power_reserve)
    m.define_demand = pyo.Constraint(m.T, rule=define_demand)


    # 3.1 CONTRAINTES PROPRES AUX CENTRALES THERMIQUES

    def max_cap_thermal(b, p, t):
        return b.prod[p, t] <= b.n[p, t] * b.Pmax[p]

    def min_cap_thermal(b, p, t):
        return b.prod[p, t] >= b.n[p, t] * b.Pmin[p]

    def max_n_plants_thermal(b, p, t):
        return b.n[p, t] <= b.N[p]

    def start_up_rule_thermal(b, p, t):
        if t == m.T.first():
            return b.n_start_up[p, t] == b.n[p, t]
        else:
            return b.n_start_up[p, t] >= b.n[p, t] - b.n[p, m.T.prev(t)]

    def cyclical_start_up_rule_thermal(b, p, t):
        if t == m.T.first():
            return b.n_start_up[p, t] >= b.n[p, t] - n_last[p]  # valeur de la veille
        else:
            return b.n_start_up[p, t] >= b.n[p, t] - b.n[p, m.T.prev(t)]  # période précédente du jour



    m.Thermal.max_capacity = pyo.Constraint(m.Thermal.Plants, m.T, rule=max_cap_thermal)
    m.Thermal.min_capacity = pyo.Constraint(m.Thermal.Plants, m.T, rule=min_cap_thermal)
    m.Thermal.max_n_plants = pyo.Constraint(m.Thermal.Plants, m.T, rule=max_n_plants_thermal)

    if cyc:
        m.Thermal.start_up_constr = pyo.Constraint(m.Thermal.Plants, m.T, rule=cyclical_start_up_rule_thermal)
    else:
        m.Thermal.start_up_constr = pyo.Constraint(m.Thermal.Plants, m.T, rule=start_up_rule_thermal)

    # 3.3 CONTRAINTES PROPRES AUX CENTRALES HYDRO

    def max_cap_hydro(b, p, t):
        return b.prod[p, t] <= b.n[p,t]*b.Pmax[p]

    def min_cap_hydro(b, p, t):
        return b.prod[p, t] >= b.n[p,t]*b.Pmax[p]

    def n_plants_hydro(b, p, t):
        return b.n[p, t] <= b.N[p]

    def start_up_rule_hydro(b, p, t):
        if t == m.T.first():
            return b.n_start_up[p, t] == b.n[p, t]
        else:
            return b.n_start_up[p, t] >= b.n[p, t] - b.n[p, m.T.prev(t)]

    def cyclical_start_up_rule_hydro(b, p, t):
        if t == m.T.first():
            return b.n_start_up[p, t] >= b.n[p, t] - n_last[p]  # valeur de la veille
        else:
            return b.n_start_up[p, t] >= b.n[p, t] - b.n[p, m.T.prev(t)]  # période précédente du jour

    def flow_out_rule(b, p, t):
        return b.flow_out[p, t] == sum(
            b.water_consumption[p] * b.n[p, t] * m.Hours[t] for p in b.Plants)  # si la centrale hydro fonctionne, elle consomme de l'eau

    def flow_in_rule(b):
        return sum(b.flow_in[t] for t in m.T) - sum(b.flow_out[p, t] for p in b.Plants for t in m.T) == 0  # niveau d'eau à l'équilibre sur la journée  # niveau d'eau à l'équilibre sur la journée

    m.Hydro.max_capacity = pyo.Constraint(m.Hydro.Plants, m.T, rule = max_cap_hydro)
    m.Hydro.min_capacity = pyo.Constraint(m.Hydro.Plants, m.T, rule = min_cap_hydro)
    m.Hydro.flow_out_rule = pyo.Constraint(m.Hydro.Plants, m.T, rule=flow_out_rule)
    m.Hydro.flow_in_rule = pyo.Constraint(rule=flow_in_rule)
    m.Hydro.n_plants = pyo.Constraint(m.Hydro.Plants, m.T, rule=n_plants_hydro)

    if cyc:
        m.Hydro.start_up_constr = pyo.Constraint(m.Hydro.Plants, m.T, rule=cyclical_start_up_rule_hydro)
    else:
        m.Hydro.start_up_constr = pyo.Constraint(m.Hydro.Plants, m.T, rule=start_up_rule_hydro)

    # 4. OBJECTIF

    def cost_thermal():
        Cstart_thermal = sum(m.Thermal.Cstart[p] * m.Thermal.n_start_up[p, t] for p in m.Thermal.Plants for t in m.T)
        Cbase_thermal = sum(m.Thermal.Cbase[p] * m.Thermal.n[p, t] * m.Hours[t] for p in m.Thermal.Plants for t in m.T)
        Cmwh_thermal = sum(
            m.Thermal.Cmwh[p] * (m.Thermal.prod[p, t] - m.Thermal.Pmin[p] * m.Thermal.n[p, t]) * m.Hours[t] for p in
            m.Thermal.Plants for t in m.T)

        return Cstart_thermal + Cbase_thermal + Cmwh_thermal

    def cost_hydro():
        Cstart_hydro = sum(m.Hydro.Cstart[p] * m.Hydro.n_start_up[p, t] for p in m.Hydro.Plants for t in m.T)
        Cbase_hydro = sum(
            m.Hydro.Cbase[p] * m.Hydro.n[p, t] * m.Hours[t] for p in m.Hydro.Plants for t in m.T)

        return Cstart_hydro + Cbase_hydro

    def total_cost(m):
        return cost_hydro() + cost_thermal()

    m.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)

    return m

