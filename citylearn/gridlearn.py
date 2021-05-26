import pandapower as pp
from pandapower import runpp
from pandapower.plotting import simple_plotly, pf_res_plotly
import pandapower.networks as networks
from citylearn import CityLearn
from citylearn import Building
from citylearn import RBC_Agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
from pettingzoo import ParallelEnv
import os

class GridLearn: # not a super class of the CityLearn environment
    def __init__(self, data_path, climate_zone, buildings_states_actions_file, hourly_timesteps, save_memory = True, building_ids=None, nclusters=2, randomseed=2, max_num_houses=None):
        self.max_num_houses = max_num_houses
        self.nclusters = nclusters

        self.data_path = data_path
        self.climate_zone = climate_zone
        self.buildings_states_actions_file = buildings_states_actions_file
        print(self.buildings_states_actions_file)
        self.hourly_timesteps = hourly_timesteps
        self.save_memory = save_memory
        self.building_ids = building_ids
        random.seed(randomseed)
        np.random.seed(randomseed)

        self.name = "test"
        self.net = self.make_grid()

        self.buildings = self.add_houses(6,0.3)
        self.agents = list(self.buildings.keys())
        self.possible_agents = self.agents[:]
        self.clusters = self.set_clusters()
        self.ncluster = 0

        self.observation_spaces = {k:v.observation_space for k,v in self.buildings.items()}
        self.action_spaces = {k:v.action_space for k,v in self.buildings.items()}

        self.metadata = {'render.modes': [], 'name':"my_env"}
        self.ts = 0

    def make_grid(self):
        # make a grid that fits the buildings generated for CityLearn
        net = networks.case33bw()

        # clear the grid of old load values
        load_nodes = net.load['bus']
        res_voltage_nodes = net.bus['name'][net.bus['vn_kv'] == 12.66]
        res_load_nodes = set(load_nodes) & set(res_voltage_nodes)

        for node in res_load_nodes:
            # remove the existing arbitrary load
            net.load.drop(net.load[net.load.bus == node].index, inplace=True)
        return net

    def add_houses(self, n, pv_penetration):
        if not self.max_num_houses:
            n = 1
        houses = []
        b = 0

        # find nodes in the network with residential voltage levels and load infrastructure
        # get the node indexes by their assigned names
        # load_nodes = self.net.load['bus']
        ext_grid_nodes = set(self.net.ext_grid['bus'])
        res_voltage_nodes = set(self.net.bus['name'][self.net.bus['vn_kv'] == 12.66])
        res_load_nodes = res_voltage_nodes - ext_grid_nodes
        # print(res_load_nodes)

        buildings = {}
        for existing_node in list(res_load_nodes)[:self.max_num_houses]:
            # remove the existing arbitrary load
            self.net.load.drop(self.net.load[self.net.load.bus == existing_node].index, inplace=True)

            # add n houses at each of these nodes
            for i in range(n):
                # bid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
                bldg = Building(self.data_path, self.climate_zone, self.buildings_states_actions_file, self.hourly_timesteps, self.save_memory, self.building_ids)
                bldg.assign_bus(existing_node)
                bldg.load_index = pp.create_load(self.net, bldg.bus, 0, name=bldg.buildingId) # create a load at the existing bus
                if np.random.uniform() <= pv_penetration:
                    bldg.gen_index = pp.create_sgen(self.net, bldg.bus, 0, name=bldg.buildingId) # create a generator at the existing bus
                else:
                    bldg.gen_index = -1
                buildings[bldg.buildingId] = bldg
        return buildings

    def set_clusters(self):
        clusters = []
        for i in range(self.nclusters):
            clusters += [self.possible_agents[i::self.nclusters]]
        return clusters

    def calc_system_losses(self):
        self.system_losses += list((self.net.res_ext_grid.p_mw + self.net.res_load.p_mw.sum() - self.net.res_gen.p_mw.sum()).values)

    def calc_voltage_dev(self):
        self.voltage_dev += list(abs((self.net.res_bus['vm_pu']-1)/0.05))

    def get_rbc_cost(self):
        # Running the reference rule-based controller to find the baseline cost
        if self.cost_rbc is None:
            env_rbc = GridLearn(self.data_path, self.building_attributes, self.weather_file, self.solar_profile, self.building_ids, hourly_timesteps=self.hourly_timesteps, buildings_states_actions = self.buildings_states_actions_filename, simulation_period = self.simulation_period, cost_function = self.cost_function, central_agent = False, n_buildings_per_bus=self.n_buildings_per_bus, pv_penetration=self.pv_penetration)
            _, actions_spaces = env_rbc.get_state_action_spaces()

            #Instantiatiing the control agent(s)
            agent_rbc = RBC_Agent(env_rbc)

            state = env_rbc.reset()
            done = False
            while not done:
                action = agent_rbc.select_action([list(env_rbc.buildings.values())[0].sim_results['hour'][env_rbc.time_step]])
                next_state, rewards, done, _ = env_rbc.step(action)
                state = next_state
            self.cost_rbc = env_rbc.get_baseline_cost()

    def reset(self):
        self.system_losses = []
        self.voltage_dev = []
        return {k:self.buildings[k].reset() for k in agents}

    def state(self, agents):
        obs = {k:np.array(self.buildings[k].get_obs(self.net)) for k in agents}
        return obs

    def get_reward(self, agents):
        rewards = {k:self.buildings[k].get_reward(self.net) for k in agents}
        return rewards

    def get_done(self, agents):
        dones = {agent: False for agent in agents}
        return dones

    def get_info(self, agents):
        infos = {agent: {} for agent in agents}
        return infos

    def get_spaces(self, agents):
        actionspace = {k:self.buildings[k].action_space for k in agents}
        obsspace = {k:self.buildings[k].observation_space for k in agents}
        return actionspace, obsspace

    def step(self, action_dict):
        # update the buildings
        if self.ts > 0:
            self.agents = self.possible_agents[(self.ts%self.nclusters)::2]

        i = 0
        for agent in action_dict:
            if i == 0:
                i += 1
            self.buildings[agent].step(action_dict[agent])

        # update the grid based on updated buildings
        self.update_grid()

        # run the grid power flow
        runpp(self.net, enforce_q_lims=True)
        self.ts += 1

        obs = self.state(list(action_dict.keys()))
        return obs, self.get_reward(list(action_dict.keys())), self.get_done(list(action_dict.keys())), self.get_info(list(action_dict.keys()))

    def update_grid(self):
        for agent, bldg in self.buildings.items():
            # Assign the load in MW (from KW in CityLearn)
            self.net.load.at[bldg.load_index, 'p_mw'] = 0.9 * bldg.current_net_electricity_demand * 0.001
            self.net.load.at[bldg.load_index, 'sn_mva'] = bldg.current_net_electricity_demand * 0.001

            if bldg.gen_index > -1:
                self.net.sgen.at[bldg.gen_index, 'p_mw'] = bldg.solar_generation * np.cos(bldg.phi) * 0.001
                self.net.sgen.at[bldg.gen_index, 'q_mvar'] = bldg.solar_generation * np.sin(bldg.phi) * 0.001

class MyEnv(ParallelEnv):
    def __init__(self, grid):
        # self.set_grid(grid)
        self.agents = grid.clusters[grid.ncluster]
        grid.ncluster = (grid.ncluster + 1) % grid.nclusters
        self.possible_agents = self.agents[:]
        self.action_spaces, self.observation_spaces = grid.get_spaces(self.agents)

        self.metadata = {'render.modes': [], 'name':"my_env"}
        self.ts = 0

    def set_grid(self, grid):
        self.grid = grid

        # self.agents = self.grid.clusters[self.grid.ncluster]
        # self.grid.ncluster = (self.grid.ncluster + 1) % self.grid.nclusters
        # self.possible_agents = self.agents[:]
        # self.action_spaces, self.observation_spaces = self.grid.get_spaces(self.agents)

    def reset(self):
        print('calling reset...')
        return self.state()

    def state(self):
        return self.grid.state(self.agents)

    def get_reward(self):
        return self.grid.get_reward(self.agents)

    def get_done(self):
        return self.grid.get_done(self.agents)

    def get_info(self):
        return self.grid.get_info(self.agents)

    def step(self, action_dict):
        print(self.agents, action_dict.keys())
        return self.grid.step(action_dict)
