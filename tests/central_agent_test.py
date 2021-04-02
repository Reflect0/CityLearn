import gym
import numpy as np
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from citylearn import CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import time
from citylearn import GridLearn

from gym.envs import register

climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
config = {
    'data_path': data_path,
    'building_attributes': data_path / 'building_attributes.json',
    'buildings_states_actions': '../citylearn/buildings_state_action_space.json',
    'weather_file': data_path / 'weather_data.csv',
    'solar_profile': data_path / 'solar_generation_1kW.csv',
    'building_ids': ['Building_3'],
    'hourly_timesteps': 3,
    'central_agent':True,
    'cost_function':['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic']
}

env_name = 'MyEnv-v1'

register(id=env_name,
     entry_point='citylearn.gridlearn:GridLearn',
     max_episode_steps=8760,
     kwargs = config)

env = gym.make(env_name)

model = SAC(MlpPolicy, env, verbose=0, learning_rate=0.01, gamma=0.99, tau=3e-4, batch_size=64, learning_starts=8759)
start = time.time()
print("starting learning")
model.learn(total_timesteps=8760*7, log_interval=1000)
print(time.time()-start)

obs = env.reset()
dones = False
counter = []
print("starting evaluation")
while dones==False:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    counter.append(rewards)
env.cost()
