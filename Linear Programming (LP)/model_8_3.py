import pyomo.environ as pyo
import pandas as pd
import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)  # éviter les messages d'erreurs

periods_temp = {(0, 6): [15000, 6],  # [demande, durée de la période (pour le calcul des MWh dans l'objectif)]
                (6, 9): [30000, 3],
                (9, 15): [25000, 6],
                (15, 18): [40000, 3],
                (18, 24): [27000, 6]}

periods_discretized = {t: [periods_temp[period_old][0], 1] for period_old in periods_temp.keys() for t in range(period_old[0], period_old[1])}


data_periods = pd.DataFrame.from_dict(periods_discretized, orient='index', columns=["Demand", "Hours"])

data_thermal = {"A": ["thermal", 12, 850, 2000, 2000, 1000, 2.0, 400, 1000, 800, 1000],  # Type, N, Pmin, Pmax, Cstart, Cbase, Cmwh, ramp_up_limit, start_limit, ramp_down_limit, stop_limit
                "B": ["thermal", 10, 1250, 1750, 1000, 2600, 1.3, 600, 1500, 1200, 1500],
                "C": ["thermal", 5, 1500, 4000, 500, 3000, 3, 800, 2000, 1700, 2000]}

data_hydro = {
    "centrale": ["H1", "H1", "H1", "H1",
                 "H2", "H2", "H2", "H2"],
    "level": [1, 2, 3, 4, 1, 2, 3, 4],
    "N": [1, 1, 1, 1,
          1, 1, 1, 1],
    "Pmax": [900, 950, 1000, 1100,
             1400, 1500, 1600, 1700],
    "Pmin": [0, 0, 0, 0,
             0, 0, 0, 0],
    "water_consumption": [0.31, 0.33, 0.35, 0.38,
                          0.47, 0.50, 0.53, 0.56],
    "elec_consumption": [3000, 3000, 3000, 3000,
                         3000, 3000, 3000, 3000],
    "Cbase": [90, 95, 105, 120,
              150, 165, 185, 210],
    "Cstart": [1500, 1500, 1500, 1500, 1200, 1200, 1200, 1200]
}

data_thermal = pd.DataFrame.from_dict(data_thermal, orient='index',
                                      columns=["Type", "N", "Pmin", "Pmax", "Cstart", "Cbase", "Cmwh", "RampUpMax", "RampStart", "RampDownMax", "RampStop"])
data_hydro = pd.DataFrame.from_dict(data_hydro).set_index(["centrale", "level"])


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

    m.Thermal.Units = pyo.Set(m.Thermal.Plants, initialize=lambda b, p: range(1, int(b.N[p]) + 1))
    m.Thermal.PU = pyo.Set(dimen=2, initialize=lambda b: [(p, u) for p in b.Plants for u in b.Units[p]])

    m.Thermal.prod = pyo.Var(m.Thermal.PU, m.T, within=pyo.NonNegativeReals)
    m.Thermal.on = pyo.Var(m.Thermal.PU, m.T, within=pyo.Binary)
    m.Thermal.n = pyo.Var(m.Thermal.Plants, m.T, within=pyo.NonNegativeIntegers)
    m.Thermal.n_start_up = pyo.Var(m.Thermal.Plants, m.T, within=pyo.NonNegativeIntegers)

    # 2. CENTRALES HYDRAULIQUES

    m.Hydro = pyo.Block()

    m.Hydro.Plants = pyo.Set(initialize=df_hydro.index.get_level_values(0).unique().tolist(), ordered=True)
    m.Hydro.PlantLevels = pyo.Set(dimen=2, initialize=list(df_hydro.index.to_list()), ordered=True)

    m.Hydro.Pmax = pyo.Param(m.Hydro.PlantLevels, initialize=df_hydro["Pmax"].to_dict(), mutable=False)
    m.Hydro.Pmin = pyo.Param(m.Hydro.PlantLevels, initialize=df_hydro["Pmin"].to_dict(), mutable=False)
    m.Hydro.water_consumption = pyo.Param(m.Hydro.PlantLevels, initialize=df_hydro["water_consumption"].to_dict(),
                                          mutable=False)
    m.Hydro.Cbase = pyo.Param(m.Hydro.PlantLevels, initialize=df_hydro["Cbase"].to_dict(), mutable=False)
    m.Hydro.Cstart = pyo.Param(m.Hydro.PlantLevels, initialize=df_hydro["Cstart"].to_dict(), mutable=False)

    m.Hydro.Elec_pump = pyo.Param(initialize=df_hydro["elec_consumption"].iloc[0])

    m.Hydro.prod = pyo.Var(m.Hydro.Plants, m.T, within=pyo.NonNegativeReals)
    m.Hydro.n = pyo.Var(m.Hydro.Plants, m.T, within=pyo.NonNegativeIntegers)
    m.Hydro.n_start_up = pyo.Var(m.Hydro.Plants, m.T,
                                 within=pyo.NonNegativeIntegers)  # nombres de centrales nouvellement allumées à chaque pas de temps
    m.Hydro.y = pyo.Var(m.Hydro.PlantLevels, m.T,
                        within=pyo.Binary)  # y[p, l, t] = 1 si la centrale hydro p tourne au niveau l à la période t, 0 sinon

    m.Hydro.flow_in = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.Hydro.flow_out = pyo.Var(m.Hydro.Plants, m.T, within=pyo.NonNegativeReals)

    m.Blocks = [m.Thermal, m.Hydro]

    # 3. CONTRAINTES

    # 3.0 CONTRAINTES GLOBALES

    def define_demand(m, t):
        return m.total_demand[t] >= m.Demand[t] + m.Hydro.flow_in[t] * m.Hydro.Elec_pump / m.Hours[
            t]  # la consommation pour remonter le niveau d'eau se rajoute à la demande totale

    def meet_demand(m, t):
        prod_thermal = sum(m.Thermal.prod[p, k, t] for p in m.Thermal.Plants for k in m.Thermal.Units[p])
        prod_hydro = sum(m.Hydro.prod[p, t] for p in m.Hydro.Plants)

        return prod_thermal + prod_hydro >= m.total_demand[t]

    def power_reserve(m, t):
        margin = 0.15

        margin_thermal = sum(
            m.Thermal.Pmax[p] * m.Thermal.n[p, t] - sum(m.Thermal.prod[p, k, t] for k in m.Thermal.Units[p]) for p in
            m.Thermal.Plants)
        margin_hydro = sum(m.Hydro.Pmax[p, 4] * (1 - m.Hydro.y[p, 4, t]) for p in m.Hydro.Plants)

        return margin_thermal + margin_hydro >= margin * m.total_demand[t]  # la somme des centrales qui produisent déjà doit pouvoir encaisser une hausse de 15% de la demande

    m.meet_demand = pyo.Constraint(m.T, rule=meet_demand)
    m.power_reserve = pyo.Constraint(m.T, rule=power_reserve)

    # 3.1 CONTRAINTES PROPRES AUX CENTRALES THERMIQUES

    def max_cap_thermal(b, p, k, t):
        return b.prod[p, k, t] <= b.Pmax[p] * b.on[p, k, t]

    def min_cap_thermal(b, p, k, t):
        return b.prod[p, k, t] >= b.Pmin[p] * b.on[p, k, t]

    def n_plants_thermal(b, p, t):
        return b.n[p, t] == sum(b.on[p, k, t] for (pp, k) in b.PU if pp == p)

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

    def ramp_up_limit(b, p, k, t):
        M = b.Pmax[p]
        return b.prod[p, k, t+1] - b.prod[p, k, t] <= b.RampUpMax[p] + M * (2 - b.on[p, k, t] - b.on[p, k, t+1])

    def ramp_down_limit(b, p, k, t):
        M = b.Pmax[p]
        return b.prod[p, k, t] - b.prod[p, k, t+1] <= b.RampDownMax[p] + M * (2 - b.on[p, k, t] - b.on[p, k, t+1])

    def ramp_start(b, p, k, t):
        M = b.Pmax[p]
        return b.prod[p, k, t+1]-b.prod[p, k, t] <= b.RampStart[p] + M * (1 - b.on[p, k, t+1] + b.on[p, k, t])

    def ramp_stop(b, p, k, t):
        M = b.Pmax[p]
        return b.prod[p, k, t]-b.prod[p, k, t+1] <= b.RampStop[p] + M * (1 - b.on[p, k, t] + b.on[p, k, t+1])

    
    
    m.Thermal.max_capacity = pyo.Constraint(m.Thermal.PU, m.T, rule=max_cap_thermal)
    m.Thermal.min_capacity = pyo.Constraint(m.Thermal.PU, m.T, rule=min_cap_thermal)
    m.Thermal.n_plants = pyo.Constraint(m.Thermal.Plants, m.T, rule=n_plants_thermal)
    m.Thermal.ramp_up_limit = pyo.Constraint(m.Thermal.PU, list(m.T)[:-1], rule=ramp_up_limit)
    m.Thermal.ramp_down_limit = pyo.Constraint(m.Thermal.PU, list(m.T)[:-1], rule=ramp_down_limit)
    m.Thermal.ramp_start = pyo.Constraint(m.Thermal.PU, list(m.T)[:-1], rule=ramp_start)
    m.Thermal.ramp_stop = pyo.Constraint(m.Thermal.PU, list(m.T)[:-1], rule=ramp_stop)


    if cyc:
        m.Thermal.start_up_constr = pyo.Constraint(m.Thermal.Plants, m.T, rule=cyclical_start_up_rule_thermal)
    else:
        m.Thermal.start_up_constr = pyo.Constraint(m.Thermal.Plants, m.T, rule=start_up_rule_thermal)

    # 3.3 CONTRAINTES PROPRES AUX CENTRALES HYDRO

    def max_cap_hydro(b, p, t):
        return b.prod[p, t] <= sum(b.Pmax[p, l] * b.y[p, l, t] for (pp, l) in b.PlantLevels if pp == p)

    def min_cap_hydro(b, p, t):
        return b.prod[p, t] >= sum(b.Pmin[p, l] * b.y[p, l, t] for (pp, l) in b.PlantLevels if pp == p)

    def n_plants_hydro(b, p, t):
        return b.n[p, t] == sum(b.y[p, l, t] for (pp, l) in b.PlantLevels if pp == p)

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
            b.water_consumption[p, l] * b.y[p, l, t] * m.Hours[t] for (pp, l) in b.PlantLevels if
            pp == p)  # si la centrale hydro fonctionne, elle consomme de l'eau

    def flow_in_rule(b):
        return sum(b.flow_in[t] for t in m.T) == sum(
            b.flow_out[p, t] for p in b.Plants for t in m.T)  # niveau d'eau à l'équilibre sur la journée

    def one_level_only(b, p, t):
        return sum(b.y[p, l, t] for (pp, l) in b.PlantLevels if pp == p) <= 1

    def prod_def(b, p, t):
        return b.prod[p, t] == sum(b.Pmax[p, l] * b.y[p, l, t] for (pp, l) in b.PlantLevels if pp == p)

    def no_pump_while_generating(b, t):
        M_in = sum(m.Hydro.water_consumption[p, 4] * m.Hours[t] for p in b.Plants for t in m.T)
        return b.flow_in[t] <= M_in * (
                    1 - sum(b.y[p, l, t] for p in b.Plants for (pp, l) in b.PlantLevels if p == pp))


    # m.Hydro.max_capacity = pyo.Constraint(m.Hydro.Plants, m.T, rule = max_cap_hydro)
    # m.Hydro.min_capacity = pyo.Constraint(m.Hydro.Plants, m.T, rule = min_cap_hydro)
    m.Hydro.n_plants = pyo.Constraint(m.Hydro.Plants, m.T, rule=n_plants_hydro)

    if cyc:
        m.Hydro.start_up_constr = pyo.Constraint(m.Hydro.Plants, m.T, rule=cyclical_start_up_rule_hydro)
    else:
        m.Hydro.start_up_constr = pyo.Constraint(m.Hydro.Plants, m.T, rule=start_up_rule_hydro)

    m.Hydro.flow_out_rule = pyo.Constraint(m.Hydro.Plants, m.T, rule=flow_out_rule)
    m.Hydro.flow_in_rule = pyo.Constraint(rule=flow_in_rule)
    m.define_demand = pyo.Constraint(m.T, rule=define_demand)
    m.Hydro.one_level_only = pyo.Constraint(m.Hydro.Plants, m.T, rule=one_level_only)
    m.Hydro.prod_def = pyo.Constraint(m.Hydro.Plants, m.T, rule=prod_def)
    m.Hydro.no_pump_while_generating = pyo.Constraint(m.T, rule=no_pump_while_generating)

    # 4. OBJECTIF

    def cost_thermal():
        Cstart_thermal = sum(m.Thermal.Cstart[p] * m.Thermal.n_start_up[p, t] for p in m.Thermal.Plants for t in m.T)
        Cbase_thermal = sum(m.Thermal.Cbase[p] * m.Thermal.n[p, t] * m.Hours[t] for p in m.Thermal.Plants for t in m.T)
        Cmwh_thermal = sum(
            m.Thermal.Cmwh[p] * (m.Thermal.prod[p, k, t] - m.Thermal.Pmin[p] * m.Thermal.on[p, k, t]) * m.Hours[t] for p
            in m.Thermal.Plants for k in m.Thermal.Units[p] for t in m.T)

        return Cstart_thermal + Cbase_thermal + Cmwh_thermal

    def cost_hydro():
        Cstart_hydro = sum(m.Hydro.Cstart[p, 1] * m.Hydro.n_start_up[p, t] for p in m.Hydro.Plants for t in m.T)
        Cbase_hydro = sum(
            m.Hydro.Cbase[p, l] * m.Hydro.y[p, l, t] * m.Hours[t] for (p, l) in m.Hydro.PlantLevels for t in m.T)

        return Cstart_hydro + Cbase_hydro

    def total_cost(m):
        return cost_hydro() + cost_thermal()

    m.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)

    return m
