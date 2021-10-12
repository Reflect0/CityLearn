import os
import numpy as np
import random
import copy
import gym
import json
import time
import math
from agents import Base_Agent

class Do_Nothing_Agent(Base_Agent):
    def __init__(self, env):
        super().__init__(env)

    def get_house_action(self, states, uid):
        return [0.0 for _ in range(self.env.action_spaces[uid].shape[0])]

    def select_action(self, states):
        action_dict = {}
        for uid in self.env.buildings.keys():
            action_dict[uid] = self.get_house_action(states, uid)

        return action_dict

class Randomized_Agent(Base_Agent):
    def __init__(self, env):
        super().__init__(env)

    def get_house_action(self, states, uid):
        return self.env.action_spaces[uid].sample()

    def select_action(self, states):
        action_dict = {}
        for uid in self.env.buildings.keys():
            action_dict[uid] = self.get_house_action(states, uid)

        return action_dict
