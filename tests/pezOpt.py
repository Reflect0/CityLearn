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
import time
from stable_baselines3.common.callbacks import CheckpointCallback
import supersuit as ss
# from ray import tune
# from ray.tune.suggest.optuna import OptunaSearch
import optuna
import os
# import ray
from pathlib import Path
# from ray.tune.suggest import ConcurrencyLimiter

# def create_envs():
climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = '../citylearn/buildings_state_action_space.json'

config = {
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "percent_rl":0.25,
    # "percent_rl":1,
    "nclusters":2,
    "max_num_houses":None
    # "max_num_houses":4
}

grid = GridLearn(**config)

envs = [MyEnv(grid), MyEnv(grid)]

print('padding action/observation spaces...')
envs = [ss.pad_action_space_v0(env) for env in envs]
envs = [ss.pad_observations_v0(env) for env in envs]

print('creating pettingzoo env...')
envs = [ss.pettingzoo_env_to_vec_env_v0(env) for env in envs]

print('stacking vec env...')
nenvs = 2
envs = [ss.concat_vec_envs_v0(env, nenvs, num_cpus=1, base_class='stable_baselines3') for env in envs]

from copy import deepcopy
grid2 = deepcopy(grid)

grids = [grid, grid2]

print('setting the grid...')
for env in envs:
    for n in range(nenvs):
        env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.grid = grids[n]
        env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.initialize_rbc_agents()

def create_trained_models(envs, params):
    models = [PPO(MlpPolicy, env, verbose=2, gamma=0.999, batch_size=512, n_steps=1, ent_coef=0.05, learning_rate=0.00001, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4) for env in envs]
    for ts in range(8760):
        for model in models:
            # print("CALL LEARN")
            model.learn(1, reset_num_timesteps=False)
    return models

def eval_models(models, ntimesteps=5):
    total_reward = 0
    obss = [env.reset() for env in envs]
    for ts in range(ntimesteps):
        for m in range(len(models)): # again, alternate through models

            # get the current observation from the perspective of the active team
            # this can probably be cleaned up
            foo = []
            for e in range(nenvs):
                bar = list(envs[m].venv.vec_envs[n].par_env.aec_env.env.env.env.env.state().values())
                for i in range(len(bar)):
                    while len(bar[i]) < 19:
                        bar[i] = np.append(bar[i], 0)
                foo += bar

            obss[m] = np.vstack(foo)

            action = models[m].predict(obss[m])[0] # send it to the SB model to select an action
            obss[m], reward, done, info = envs[m].step(action) # update environment
            total_reward += np.sum(reward)
    return total_reward

def objective(trial):

    # joblib.dump(study, 'study.pkl')

    params = {
        "n_epochs": trial.suggest_int('n_epochs', 1, 50),
        "gamma": trial.suggest_uniform('gamma', .9, .999),
        "ent_coef": trial.suggest_loguniform('ent_coef', .001, .1),
        "learning_rate": trial.suggest_loguniform('learning_rate', 5e-6, 5e-4),
        "vf_coef": trial.suggest_uniform('vf_coef', .1, 1),
        "gae_lambda": trial.suggest_uniform('gae_lambda', .8, 1),
        "max_grad_norm": trial.suggest_loguniform('max_grad_norm', .01, 10),
        "n_steps": trial.suggest_categorical('n_steps', [1]),
        "batch_size": trial.suggest_categorical('batch_size', [1024, 4096, 8192]),  # , 512, 1024, 2048, 4096
        # "n_envs": trial.suggest_categorical('n_envs', [2, 4, 8]),
        "clip_range": trial.suggest_uniform('clip_range', .1, 5)
    }

    models = create_trained_models(envs, params)

    return -1 * eval_models(models, 720)

study = optuna.create_study()
study.optimize(objective, timeout= 3600*10) # timeout 10 hours
print(study.best_params)
