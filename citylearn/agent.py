import os
import numpy as np
import random
import copy
import gym
import json
import time
import math

class Base_Agent:
    """ A base agent for RL learning. Agent works on the community level to get
    actions for each home energy system (HVAC, Water Heater, PV). """
    def __init__(self, env):
        self.env = env
        self.action_tracker = self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    def get_house_action(self, states, uid):
        """ Gets the actions on the house-level """
        pass

    def select_action(self, states):
        """ Analogous to model.predict() in stable baselines.
        Order of action list = [HVAC, Water Heater, PV]
        Returns a dictionary {building unique id : [list of actions]}"""
        pass

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


class RBC_Agent:
    def __init__(self, env):
        self.env = env
        self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    def predict(self):
        hour_day = self.env.time_step % (self.env.hourly_timesteps * 24)
        daytime = True if hour_day >= 1 and hour_day <= 21 else False
        actions = []
        for action, enabled in self.env.enabled_actions.items():
            if enabled:
                if action == 'cooling_storage' or action == 'dhw_storage' or action == 'electrical_storage':
                    if daytime:
                        actions += [0.0]#[-.08]
                    else:
                        actions += [0.0]#[0.91]
                elif action == 'pv_curtail':
                    actions += [-1.0]
                else:
                    actions += [0.0]

        return actions # casting this as a list of list matches the predict function in Stable Baselines.

# this class is used to replace the RL agents in the grid directly
class RBC_Agent_v2:
    def __init__(self, env):
        self.env = env
        self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    def predict(self):
        hour_day = self.env.time_step % (self.env.hourly_timesteps * 24)
        daytime = True if hour_day >= 1 and hour_day <= 21 else False
        actions = []
        for action, enabled in self.env.enabled_actions.items():
            if enabled:
                if action == 'cooling_storage' or action == 'dhw_storage' or action=='electrical_storage':
                    if daytime:
                        actions += [0.0]#[-0.08]
                    else:
                        actions += [0.0]#[0.91]
                elif action == 'pv_curtail':
                    actions += [-1.0]
                else:
                    actions += [0.0]

        return actions
