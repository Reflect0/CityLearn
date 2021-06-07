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
import supersuit as ss

import time
import os

model_name = 'randomized_buses'

tic = time.time()
# multiprocessing.set_start_method("fork")

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
    "nclusters":2,
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

from copy import deepcopy
# grid2 = deepcopy(grid)
#
# grids = [grid, grid2]
grids = [grid]
grids += [deepcopy(grid) for _ in range(nenvs)]

print('setting the grid...')
for env in envs:
    for n in range(nenvs):
        env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.grid = grids[n]
        env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.initialize_rbc_agents()

models = [PPO(MlpPolicy, env, verbose=2, gamma=0.999, batch_size=512, n_steps=1, ent_coef=0.0001, learning_rate=0.0001, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95) for env in envs]

for ts in range(8760*2):
    for model in models:
        # print("CALL LEARN")
        model.learn(1, reset_num_timesteps=False)
if not os.path.exists(f'models/{model_name}'):
    os.makedirs(f'models/{model_name}')
os.chdir(f'models/{model_name}')
for m in range(len(models)):
    models[m].save(f"model_{m}")

toc = time.time()
print(toc-tic)
