import os
import numpy as np
import random
import copy
import gym
import json
import time
import math
from agents import Base_Agent

class RBC_Agent(Base_Agent):
    def __init__(self, env):
        super.__init__(env)

    def reset_action_tracker(self):
        self.action_tracker = []

    def get_tou_storage(self, hour):
        if hour < 7:
            a = 0.1383
        elif hour < 16:
            a = -0.05
        elif hour < 18:
            a = -0.11
        elif hour < 22:
            a = -0.06
        else:
            a = 0.085
        return a

    def predict(self):
        hour_day = self.env.time_step / self.env.hourly_timesteps % 24
        tou_storage = self.get_tou_storage(hour_day)
        daytime = True if hour_day >= self.env.morning and hour_day <= self.env.night else False

        actions = []
        if self.env.enabled_actions['cooling_storage']:
            actions += [tou_storage]

        if self.env.enabled_actions['dhw_storage']:
            actions += [tou_storage]

        if self.env.enabled_actions['pv_curtail']:
            actions += [1.0]

        if self.env.enabled_actions['pv_phi']:
            actions += [-1.0]

        if self.env.enabled_actions['electrical_storage']:
            actions += [0]

        return actions

# this class is used to replace the RL agents in the grid directly
class RBC_Agent_v2(RBC_Agent):
    def __init__(self, env):
        super().__init__(env)

    def predict(self):
        hour_day = self.env.time_step/self.env.hourly_timesteps % 24
        tou_storage = self.get_tou_storage(hour_day)
        daytime = True if hour_day >= self.env.morning and hour_day <= self.env.night else False
        actions = []
        if self.env.enabled_actions['cooling_storage']:
            actions += [tou_storage]

        if self.env.enabled_actions['dhw_storage']:
            actions += [tou_storage]

        if self.env.enabled_actions['pv_curtail']:
            actions += [1.0]

        if self.env.enabled_actions['pv_phi']:
            actions += [-1.0]

        if self.env.enabled_actions['electrical_storage']:
            actions += [0]

        return actions
