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

def train_models(models, model_name, n_steps=8759*4):
    toc = time.time()
    for ts in range(n_steps):
        for model in models:
            model.learn(1, reset_num_timesteps=False)

    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')
    os.chdir(f'models/{model_name}')
    for m in range(len(models)):
        models[m].save(f"model_{m}")
    tic = time.time()
    print(f"Training done in {tic-toc} seconds")
    return
