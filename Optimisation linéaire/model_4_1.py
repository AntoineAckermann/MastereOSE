import pyomo.environ as pyo
import pandas as pd
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR) # éviter les messages d'erreur

data_thermal = {"A": [12, 850, 2000, 2000, 1000, 2.0], # N, Pmin, Pmax, Cmwh
         "B": [10, 1250, 1750, 1000, 2600, 1.3],
         "C": [5, 1500, 4000, 500, 3000, 3.0]}

periods = {"0h-6h": [15000, 6], # [demande, durée de la période (pour le calcul des MWh dans l'objectif)]
          "6h-9h": [30000, 3],
          "9h-15h": [25000, 6],
          "15h-18h": [40000, 3], 
          "18h-24h": [27000, 6]}



data_thermal = pd.DataFrame.from_dict(data_thermal, orient='index', columns=["N", "Pmin", "Pmax", "Cstart", "Cbase", "Cmwh"])
data_periods = pd.DataFrame.from_dict(periods, orient='index', columns=["Demand", "Hours"])


def build_model(df_plants, df_periods, opt, cyc=True):
    
    if cyc:
        non_cyc_model = build_model(df_plants, df_periods, opt, cyc=False) # il faut d'abord faire tourner le modèle avec planification non-cyclique pour obtenir les dernières valeurs pour les intégrer dans le modèle cyclique
        opt.solve(non_cyc_model)
        last_t = non_cyc_model.Periods.last()
        n_last = {p: pyo.value(non_cyc_model.n[p, last_t]) for p in non_cyc_model.Plants}

    
    # 0. MODELE GLOBAL

    model = pyo.ConcreteModel("Daily planning")
    
    model.Plants = pyo.Set(initialize = df_plants.index.tolist(), ordered=True)
    model.Periods = pyo.Set(initialize = df_periods.index.tolist(), ordered=True)
    
    for col in df_plants.columns: # pour chaque caractéristique N, Pmin, Pmax, Cmwh associée à chaque technologie
        setattr(model, col, pyo.Param(model.Plants, initialize=df_plants[col].to_dict()))
    
    for col in df_periods.columns: # pour chaque caractéristique demande, durée associée à chaque période
        setattr(model, col, pyo.Param(model.Periods, initialize=df_periods[col].to_dict()))
    
    # 1. DEFINITION DES VARIABLES
    
    model.prod = pyo.Var(model.Plants, model.Periods, within=pyo.NonNegativeReals) # production de chaque centrale à chaque pas de temps
    model.n = pyo.Var(model.Plants, model.Periods, within=pyo.NonNegativeIntegers) # nombres de centrales allumées de chaque type
    model.n_start_up = pyo.Var(model.Plants, model.Periods, within=pyo.NonNegativeIntegers)
    
    
    # 2. DEFINITION DES CONTRAINTES
    
    def meet_demand(m, period):
        return sum(m.prod[p, period] for p in m.Plants) >= m.Demand[period]
    
    def max_cap(m, plant, period):
        return m.prod[plant, period] <= m.n[plant, period] * m.Pmax[plant]
    
    def min_cap(m, plant, period):
        return m.prod[plant, period] >= m.n[plant, period] * m.Pmin[plant]
    
    def max_n_plants(m, plant, period):
        return m.n[plant, period] <= m.N[plant]
    
    
    def power_reserve(m, period):
        margin = 0.15
        return sum( m.Pmax[plant]*m.n[plant, period] - m.prod[plant, period] for plant in m.Plants ) >=  margin * m.Demand[period] 

    def start_up_rule(m, p, t):
        if t == m.Periods.first():
            return m.n_start_up[p, t] == m.n[p, t]
        else:
            return m.n_start_up[p, t] >= m.n[p, t] - m.n[p, m.Periods.prev(t)]

    def cyclical_start_up_rule(m, p, t):
        if t == m.Periods.first():
            return m.n_start_up[p, t] >= m.n[p, t] - n_last[p]  # valeur de la veille
        else:
            return m.n_start_up[p, t] >= m.n[p, t] - m.n[p, m.Periods.prev(t)]  # période précédente du jour

    if cyc:
        model.start_up_constr = pyo.Constraint(model.Plants, model.Periods, rule=cyclical_start_up_rule)
    else:
        model.start_up_constr = pyo.Constraint(model.Plants, model.Periods, rule=start_up_rule)
    
    model.meet_demand = pyo.Constraint(model.Periods, rule=meet_demand)
    model.max_capacity = pyo.Constraint(model.Plants, model.Periods, rule = max_cap)
    model.min_capacity = pyo.Constraint(model.Plants, model.Periods, rule = min_cap)
    model.max_n_plants = pyo.Constraint(model.Plants, model.Periods, rule = max_n_plants)
    model.power_reserve = pyo.Constraint(model.Periods, rule=power_reserve)

    # 3. DEFINITION DE l'OBJECTIF
    
    def total_cost():
        total_cost = 0

        for plant in model.Plants:
            for period in model.Periods:
                total_cost += model.Cstart[plant] * model.n_start_up[plant, period] + ( model.Cbase[plant] * model.n[plant, period] + model.Cmwh[plant] * ( model.prod[plant, period] - model.Pmin[plant] * model.n[plant, period] ) ) * model.Hours[period]
        
        return total_cost
    
    model.obj = pyo.Objective(expr=total_cost(), sense=pyo.minimize)
    
    return model

