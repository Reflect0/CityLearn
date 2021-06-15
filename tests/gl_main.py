# The main.py for the gridlearn fork of citylearn.

import pandapower as pp
from pandapower import runpp
from pandapower.plotting import simple_plotly, pf_res_plotly
import pandapower.networks as networks
from citylearn import GridLearn, Do_Nothing_Agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

np.random.seed(0)
climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = '../citylearn/buildings_state_action_space.json'
# building_ids = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9","Building_10"]
building_ids = ["Building_1","Building_4","Building_5","Building_6"] # only building types with pv
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']
hourly_steps = 6
my_grid = GridLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, hourly_steps, buildings_states_actions = building_state_actions, cost_function = objective_function, verbose=1, n_buildings_per_bus=1)

observations_spaces, actions_spaces = my_grid.get_state_action_spaces()

# Simulation without energy storage
my_grid.reset()
done = False

agent = Do_Nothing_Agent(my_grid)

# while not done:
print("Running power flows...")
state = my_grid.reset()
for ts in range(6*24):
    # for the do nothing action:
    action = agent.select_action(state)
    state, rewards, done, _ = my_grid.step(action)

print("Plotting results...")
pf_res_plotly(my_grid.net)
print()

my_grid.plot_all()
