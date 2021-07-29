from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import multiprocessing
from pettingzoo.test import parallel_api_test
from citylearn import GridLearn
from citylearn import MyEnv
from pathlib import Path
# from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3 import PPO
import gym
import numpy as np
import supersuit as ss
from copy import deepcopy
import time
from ray.tune.registry import register_env
from ray.rllib.agents.registry import get_trainer_class
import os
import ray

def create_env(cls, args):
    grid = GridLearn(**args)
    env = cls(grid)
    env.grid = grid
    env = ss.pad_action_space_v0(env)
    env = ss.pad_observations_v0(env)
    env = ParallelPettingZooEnv(env)
    return env

climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = '../citylearn/buildings_state_action_space.json'

grid_config = {
    "model_name":'test',
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "max_num_houses":None,
    "percent_rl":0.25,
    "nclusters":1,
    "save_memory":False
}

grid = GridLearn(**grid_config)
env = MyEnv(grid)

register_env("simple_grid", lambda args: create_env(MyEnv, args))

num_cpus = 2
num_rollouts = 2
alg_name = "PPO"
# Gets default training configuration and specifies the POMgame to load.
config = deepcopy(get_trainer_class(alg_name)._default_config)

config["env_config"] = grid_config

# Configuration for multiagent setup with policy sharing:
config["multiagent"] = {
    # Setup a single, shared policy for all agents.
    "policies": {k:(None, v.observation_space, v.action_space, {}) for k,v in grid.buildings.items()},
    # Map all agents to that policy.
    "policy_mapping_fn": lambda agent_id, **kwargs: agent_id,
}

# Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
config["log_level"] = "DEBUG"
config["num_workers"] = 4
# Fragment length, collected at once from each worker and for each agent!
config["rollout_fragment_length"] = 30
# Training batch size -> Fragments are concatenated up to this point.
config["train_batch_size"] = 200
# After n steps, force reset simulation
config["horizon"] = 8760*4
# Default: False
config["no_done_at_end"] = False
# Info: If False, each agents trajectory is expected to have
# maximum one done=True in the last step of the trajectory.
# If no_done_at_end = True, environment is not resetted
# when dones[__all__]= True.

# Initialize ray and trainer object
ray.init(num_cpus=num_cpus + 1, ignore_reinit_error=True, object_store_memory=2*10**9)
trainer = get_trainer_class(alg_name)(env="simple_grid", config=config)

# Train once
trainer.train()

ep_reward = 0
obs = env.reset()
for _ in range(5):
    action = {}
    for agent_id, agent_obs in obs.items():
        policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
        action[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
    #print(action)
    obs, reward, done, info = env.step(action)
    ep_reward += sum(reward.values())
