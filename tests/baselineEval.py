import multiprocessing
import sys
from pettingzoo.test import parallel_api_test

import multiprocessing
import sys
from pettingzoo.test import parallel_api_test
from citylearn import GridLearn
from citylearn import MyEnv
from pathlib import Path
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import gym
import numpy as np

import multiprocessing
import sys
import supersuit as ss

import time
import os

climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = '../citylearn/buildings_state_action_space.json'

config = {
    "model_name":"baseline_3xsolar",
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "percent_rl":0.1,
    # "percent_rl":1,
    "nclusters":4,
    "max_num_houses":None
    # "max_num_houses":4
}

grid = GridLearn(**config)

envs = [MyEnv(grid) for _ in range(config['nclusters'])]

print('setting the grid...')
for env in envs:
    env.grid = grid
    # env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.initialize_rbc_agents()
    env.initialize_rbc_agents('all')

# models = [PPO.load(f"models/10_houses/model_{m}") for m in range(len(envs))]

sum_reward = 0
obss = [env.reset() for env in envs]
for ts in range(51*7*24*4): # test on 5 timesteps
    for m in range(len(envs)): # again, alternate through models

        obss[m], reward, done, info = envs[m].step({}) # update environment

for building in grid.rl_agents:
    print(building, grid.buildings[building].bus)

grid.plot_all()
