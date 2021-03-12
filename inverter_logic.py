import pandapower as pp
from pandapower import runpp
from pandapower.plotting import simple_plotly, pf_res_plotly
import pandapower.networks as networks
from citylearn import CityLearn
from gridlearn import GridLearn
from agent import RBC_Agent, Do_Nothing_Agent, Randomized_Agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ["Building_1"]
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
hourly_steps = 2
print("Initializing the grid...")
my_grid = GridLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, hourly_steps, buildings_states_actions = building_state_actions, cost_function = objective_function, verbose=1, n_buildings_per_bus=1, pv_penetration=1, test=True)

# Simulation without energy storage
state = my_grid.reset()
done = False

agent = Do_Nothing_Agent(my_grid)
# agent = RBC_Agent(my_grid)
for ts in range(96):
    print(ts, my_grid.time_step)
    action = agent.select_action(state)
    state, rewards, done, _ = my_grid.step(action)

print("Plotting results...")
pf_res_plotly(my_grid.net)

my_grid.plot_all()
