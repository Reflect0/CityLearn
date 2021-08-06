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
from copy import deepcopy
import multiprocessing
import sys
import supersuit as ss

import time
import os

# multiprocessing.set_start_method("fork")
model_name = "pv_rec"

climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = '../citylearn/buildings_state_action_space.json'

config = {
    "model_name":model_name,
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

print('padding action/observation spaces...')
envs = [ss.pad_action_space_v0(env) for env in envs]
envs = [ss.pad_observations_v0(env) for env in envs]

print('creating pettingzoo env...')
envs = [ss.pettingzoo_env_to_vec_env_v0(env) for env in envs]

print('stacking vec env...')
nenvs = 2
envs = [ss.concat_vec_envs_v0(env, nenvs, num_cpus=1, base_class='stable_baselines3') for env in envs]

grids = [grid]
grids += [deepcopy(grid) for _ in range(nenvs-1)]

print('setting the grid...')
for env in envs:
    for n in range(nenvs):
        env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.grid = grids[n]
        env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.initialize_rbc_agents()

models = [PPO.load(f"models/{model_name}/model_{m}") for m in range(len(envs))]

sum_reward = 0
obss = [env.reset() for env in envs]
for ts in range(7*24*4): # test on 5 timesteps
    for m in range(len(models)): # again, alternate through models

        # get the current observation from the perspective of the active team
        # this can probably be cleaned up
        foo = []
        for e in range(nenvs):
            bar = list(envs[m].venv.vec_envs[n].par_env.aec_env.env.env.env.env.state().values())
            for i in range(len(bar)):
                while len(bar[i]) < 18:
                    bar[i] = np.append(bar[i], 0)
            foo += bar

        obss[m] = np.vstack(foo)

        action = models[m].predict(obss[m])[0] # send it to the SB model to select an action
        obss[m], reward, done, info = envs[m].step(action) # update environment
#         sum_reward += np.sum(reward)
# print(sum_reward)
grid.plot_all()
