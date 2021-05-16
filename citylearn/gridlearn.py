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

class GridLearn(ParallelEnv): # not a super class of the CityLearn environment
    def __init__(self, data_path, climate_zone, buildings_states_actions_file, hourly_timesteps, save_memory = True, building_ids=None, nclusters=3, randomseed=2):
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

        self.buildings = self.add_houses(1,0.3)
        self.agents = list(self.buildings.keys())
        self.possible_agents = self.agents[:]

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
        for existing_node in res_load_nodes:
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
        return {k:v.reset(self.net) for k,v in self.buildings.items()}

    def state(self):
        # results = self.net.load.merge(self.net.res_bus, left_on='bus', right_index=True)
        # obs = results.set_index('name')['vm_pu'].to_dict()
        # obs = {k: np.array([v]) for k,v in obs.items()}
        # self.obs = obs
        obs = {k:np.array(v.get_obs(self.net)) for k,v in self.buildings.items()}
        return obs

    def get_reward(self):
        rewards = {k: v.get_reward(self.net) for k,v in self.buildings.items()}
        return rewards

    def get_done(self):
        dones = {agent: False for agent in self.agents}
        return dones

    def get_info(self):
        infos = {agent: {} for agent in self.agents}
        return infos

    def step(self, action_dict):
        # update the buildings
        # if self.ts > 0:
        #     self.agents = self.possible_agents[(self.ts%self.nclusters)::2]

        for agent in self.agents:
            self.buildings[agent].step(action_dict[agent])

        # update the grid based on updated buildings
        self.update_grid()

        # run the grid power flow
        runpp(self.net, enforce_q_lims=True)
        self.ts += 1

        obs = self.state()
        key = list(obs.keys())[0]
        print(obs[key], obs[key].shape)
        return obs, self.get_reward(), self.get_done(), self.get_info()

    def update_grid(self):
        for agent, bldg in self.buildings.items():
            # Assign the load in MW (from KW in CityLearn)
            self.net.load.at[bldg.load_index, 'p_mw'] = 0.9 * bldg.current_net_electricity_demand * 0.001
            self.net.load.at[bldg.load_index, 'sn_mva'] = bldg.current_net_electricity_demand * 0.001

            if bldg.gen_index > -1:
                self.net.sgen.at[bldg.gen_index, 'p_mw'] = bldg.solar_generation * np.cos(bldg.phi) * 0.001
                self.net.sgen.at[bldg.gen_index, 'q_mvar'] = bldg.solar_generation * np.sin(bldg.phi) * 0.001

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def plot_buses(self):
        df = self.output['vm_pu']['values']
        xfmr = set(self.net.bus.iloc[self.net.trafo.hv_bus].index) | set(self.net.bus.iloc[self.net.trafo.lv_bus].index)
        ext_grid = set(self.net.bus.iloc[self.net.ext_grid.bus].index)
        substation = xfmr | ext_grid
        loads = set(self.net.load.bus)
        buses = set(self.net.bus.index) - substation
        gens = set(self.net.gen.bus)

        # substation buses
        self.plot_northsouth([df.loc[substation]], title="Substation Voltages", y="Vm_pu")

        # generator buses
        gen_buses = gens & buses
        non_gen_buses = gens ^ buses
        if not len(gen_buses) == 0:
            self.plot_northsouth([df.loc[gen_buses]], title="Buses with PV", y="Vm_pu")

        # buses with building loads
        building_ng_buses = non_gen_buses & loads
        if not len(building_ng_buses) == 0:
            self.plot_northsouth([df.loc[building_ng_buses]], title="Building Voltages", y="Vm_pu")

        # other buses (distribution strictly)
        other_ng_buses = non_gen_buses - building_ng_buses
        if not len(other_ng_buses) == 0:
            self.plot_northsouth([df.loc[other_ng_buses]], title="Distribution buses", y="Vm_pu")

    def plot_northsouth(self, dfs, title="", y=""):
        line = self.net.bus_geodata['x'].median()

        temp = self.net.bus_geodata.merge(self.net.bus, left_index=True, right_index=True)
        north_buses = set(temp.loc[temp["x"] > line].index)
        south_buses = set(temp.loc[temp["x"] <= line].index)

        fig, axes = plt.subplots(nrows=len(dfs),ncols=2, figsize=(20,8))
        plt.subplots_adjust(hspace = 0.5, wspace=0.25)
        for i in range(len(dfs)): # can pass p and q vars
            north_list = set(dfs[i].index) & north_buses
            south_list = set(dfs[i].index) & south_buses

            if len(south_list) > 0:
                if len(dfs) > 1:
                    quad = axes[i][0]
                else:
                    quad = axes[0]

                f = dfs[i].loc[south_list].transpose().plot(ax=quad, figsize=(10,6), color=plt.cm.Spectral(np.linspace(0, 1, len(dfs[i]))))
                f.set_xlabel(f"Timestep ({60/self.hourly_timesteps} minutes)")
                f.set_ylabel(y[i])
                f.set_title(f"South, {title}")
                quad.legend().set_visible(False)

            if len(north_list) > 0:
                if len(dfs) > 1:
                    quad = axes[i][1]
                else:
                    quad = axes[1]

                g = dfs[i].loc[north_list].transpose().plot(ax=quad, figsize=(10,6), color=plt.cm.Spectral(np.linspace(1, 0, len(dfs[i]))))
                g.set_xlabel(f"Time ({60/self.hourly_timesteps} minutes)")
                g.set_ylabel(y[i])
                g.set_title(f"North, {title}")
                quad.legend().set_visible(False)

    def plot_all(self):
        self.plot_buses()
        self.plot_northsouth([self.output['p_mw_load']['values']], title="Building loads", y=["P (MW)"])
        self.plot_northsouth([self.output['p_mw_gen']['values'], self.output['q_mvar_gen']['values']], title="Generation", y=["P (MW)", "Q (MVAR)"])
        plt.show()
