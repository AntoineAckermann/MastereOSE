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


def build_model(df_plants, df_periods):

    
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
    
    model.meet_demand = pyo.Constraint(model.Periods, rule=meet_demand)
    
    def max_cap(m, plant, period):
        return m.prod[plant, period] <= m.n[plant, period] * m.Pmax[plant]
    
    def min_cap(m, plant, period):
        return m.prod[plant, period] >= m.n[plant, period] * m.Pmin[plant]
    
    def max_n_plants(m, plant, period):
        return m.n[plant, period] <= m.N[plant]
    
    def start_up_rule(m, plant, period):

        if period == m.Periods.first():
            return m.n_start_up[plant, period] == m.n[plant, period]
        else:
            return m.n_start_up[plant, period] >= m.n[plant, period] - m.n[plant, m.Periods.prev(period)]



    
    model.max_capacity = pyo.Constraint(model.Plants, model.Periods, rule = max_cap)
    model.min_capacity = pyo.Constraint(model.Plants, model.Periods, rule = min_cap)
    model.max_n_plants = pyo.Constraint(model.Plants, model.Periods, rule = max_n_plants)
    model.start_up_constr = pyo.Constraint(model.Plants, model.Periods, rule=start_up_rule)
    
    # 3. DEFINITION DE l'OBJECTIF
    
    def total_cost():
        total_cost = 0

        for plant in model.Plants:
            for period in model.Periods:
                total_cost += model.Cstart[plant] * model.n_start_up[plant, period] + ( model.Cbase[plant] * model.n[plant, period] + model.Cmwh[plant] * ( model.prod[plant, period] - model.Pmin[plant] * model.n[plant, period] ) ) * model.Hours[period]
        
        return total_cost
    
    model.obj = pyo.Objective(expr=total_cost(), sense=pyo.minimize)
    
    return model
