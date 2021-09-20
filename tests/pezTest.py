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
from copy import deepcopy
import time
import os
import random
import time
random.seed(12)
np.random.seed(12)

model_name = "test"

tic = time.time()

climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = '../citylearn/buildings_state_action_space.json'

config = {
    "model_name":model_name,
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "percent_rl":0.5,
    "nclusters":1,
    "max_num_houses":None
}

grid = GridLearn(**config)

env = MyEnv(grid) #for _ in range(config['nclusters'])
env.grid = grid
env.initialize_rbc_agents()

print('creating pettingzoo env...')
env = ss.pettingzoo_env_to_vec_env_v0(env)

print('stacking vec env...')
# nenvs = 2
env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

grid.normalize_reward()

# print('setting the grid...')
# for env in envs:
#     for n in range(nenvs):
#         # env.venv.vec_envs[n].par_env.aec_env.env.env.env.grid = grids[n]
#         env.venv.vec_envs[n].par_env.grid = grids[n]
#         # env.venv.vec_envs[n].par_env.aec_env.env.env.env.initialize_rbc_agents()
#         env.venv.vec_envs[n].par_env.initialize_rbc_agents()

model = PPO(MlpPolicy, env, ent_coef=0.1, learning_rate=0.001, n_epochs=30)

# nloops=1
# for loop in range(nloops):
env.reset()
print('==============')
models[0].learn(4*4*8759, verbose=1)
print('==============')
if not os.path.exists(f'models/{model_name}'):
    os.makedirs(f'models/{model_name}')
os.chdir(f'models/{model_name}')
for m in range(len(models)):
    print('saving trained model')
    models[m].save(f"model_{m}")
os.chdir('../..')

toc = time.time()
print(toc-tic)
