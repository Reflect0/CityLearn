# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from gridlearn import GridLearn
# import matplotlib.pyplot as plt
from pathlib import Path
from agent import RL_Agents_Coord
import numpy as np
import csv
import time
import re
import pandas as pd
import torch
from joblib import dump, load

# Load environment
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_id = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic','system_losses']

print("Initializing the grid...")
# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
# Can be obtained using observations_spaces[i].low or .high
env = GridLearn(data_path, building_attributes, weather_file, solar_profile, building_id, 6, buildings_states_actions = building_state_actions, cost_function = objective_function, verbose=1, n_buildings_per_bus=1)

# Hyperparameters
bs = 256
tau = 0.005
gamma = 0.99
lr = 0.0003
hid = [256,256]

n_episodes = 12

print("Initializing the agents...")
# Instantiating the control agent(s)
agents = RL_Agents_Coord(env, list(self.buildings.keys()), discount = gamma, batch_size = bs, replay_buffer_capacity = 1e5, regression_buffer_capacity = 12*8760, tau=tau, lr=lr, hidden_dim=hid, start_training=8760*3, exploration_period = 8760*3+1,  start_regression=8760, information_sharing = True, pca_compression = .95, action_scaling_coef=0.5, reward_scaling = 5., update_per_step = 1, iterations_as = 2)

print("Starting the experiment...")
# The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
start = time.time()
for e in range(n_episodes):
    is_evaluating = (e > 7) # Evaluate deterministic policy after 7 epochs
    rewards = []
    state = env.reset()
    done = False

    j = 0
    action, coordination_vars = agents.select_action(state, deterministic=is_evaluating)
    while not done:
        next_state, reward, done, _ = env.step(action)
        action_next, coordination_vars_next = agents.select_action(next_state, deterministic=is_evaluating)
        agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)

        state = next_state
        coordination_vars = coordination_vars_next
        action = action_next

    print('Loss -',env.cost(), 'Simulation time (min) -',(time.time()-start)/60.0)
