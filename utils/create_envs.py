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

def make_envs(grid, n_clusters, n_parallel_envs=2):
    envs = [MyEnv(grid) for _ in range(n_clusters)]

    print('padding action/observation spaces...')
    envs = [ss.pad_action_space_v0(env) for env in envs]
    envs = [ss.pad_observations_v0(env) for env in envs]

    print('creating pettingzoo env...')
    envs = [ss.pettingzoo_env_to_vec_env_v0(env) for env in envs]

    print('stacking vec env...')
    envs = [ss.concat_vec_envs_v0(env, n_parallel_envs, num_cpus=1, base_class='stable_baselines3') for env in envs]

    grids = [grid]
    grids += [deepcopy(grid) for _ in range(n_parallel_envs)]

    print('setting the grid...')
    for env in envs:
        for n in range(n_parallel_envs):
            env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.grid = grids[n]
            env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.initialize_rbc_agents()
    return envs

def make_models(envs, verbose=2, gamma=0.999, batch_size=512, n_steps=1, ent_coef='auto_0.1', learning_rate=0.0001, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95):
    models = [PPO(MlpPolicy, env, verbose=verbose, gamma=gamma, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, learning_rate=learning_rate, vf_coef=vf_coef, max_grad_norm=max_grad_norm, gae_lambda=gae_lambda) for env in envs]
    return models

def load_models(envs, names):
    models = [PPO.load(f"models/{model_name}/model_{m}") for m in range(len(envs))]
    return models
