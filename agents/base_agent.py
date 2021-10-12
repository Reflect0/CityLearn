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
        Order of action list = [HVAC, Water Heater, Inverter, Battery]
        Returns a dictionary {building unique id : [list of actions]}"""
        pass
