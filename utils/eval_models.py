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

def eval_models(models, envs, n_steps=7*27*4):
    for ts in range(n_steps):
        for m in range(len(models)): # again, alternate through models
            foo = []
            for e in range(nenvs):
                bar = list(envs[m].venv.vec_envs[e].par_env.aec_env.env.env.env.env.state().values())
                for i in range(len(bar)):
                    while len(bar[i]) < 20:
                        bar[i] = np.append(bar[i], 0)
                foo += bar

            obss[m] = np.vstack(foo)

            action = models[m].predict(obss[m])[0] # send it to the SB model to select an action
            obss[m], reward, done, info = envs[m].step(action) # update environment
    grid.plot_all()
    return
