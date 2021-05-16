import multiprocessing
import sys
from pettingzoo.test import parallel_api_test
from citylearn import GridLearn
from pathlib import Path
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import gym

import multiprocessing
import sys
import supersuit as ss

import time

if __name__=="__main__":
    climate_zone = 1
    data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
    buildings_states_actions = '../citylearn/buildings_state_action_space.json'

    config = {
        "data_path":data_path,
        "climate_zone":climate_zone,
        "buildings_states_actions_file":buildings_states_actions,
        "hourly_timesteps":4
    }

    env = GridLearn(**config)
    # parallel_api_test(env)

    try:
        multiprocessing.set_start_method("fork")
    except:
        pass

    print("shaping obs and action spaces...")
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    print("enabling death mode...")
    env = ss.black_death_v1(env)

    print("vectorizing environments to use stable baselines...")
    tic = time.time()
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 2, num_cpus=2, base_class='stable_baselines3')
    toc = time.time()
    print(f"it took {tic - toc} seconds")

    print("making models...")
    models = []
    for _ in range(3):
        models += [PPO(MlpPolicy, env, verbose=1, gamma=0.999, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1)]

    print("training models...")
    for ts in range(10):
        for _ in range(1000): # timesteps
            for model in models:
                model.learn(1, reset_num_timesteps=False)
                print(f"step {_}")
        for i in range(len(models)):
            model.save(f"model_{i}")
    # multiprocessing pool hangs but I'm not sure where to close it
    print('learning done')
    env.close()
