import pandapower as pp
from pandapower import runpp
from pandapower.plotting import simple_plotly, pf_res_plotly
import pandapower.networks as networks
from citylearn import CityLearn
from citylearn import Building, Weather
from agents import RBC_Agent, RBC_Agent_v2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
from pettingzoo import ParallelEnv
import os
import matplotlib.pyplot as plt
import json

class GridLearn: # not a super class of the CityLearn environment
    def __init__(self, model_name, data_path, climate_zone, buildings_states_actions_file, hourly_timesteps, save_memory = True, building_ids=None, nclusters=2, randomseed=2, max_num_houses=None, percent_rl=1):
        self.model_name = model_name
        if not os.path.isdir(f'models/{self.model_name}'):
            os.mkdir(f'models/{self.model_name}')
        self.max_num_houses = max_num_houses
        self.nclusters = nclusters
        self.percent_rl = percent_rl

        self.data_path = data_path
        self.climate_zone = climate_zone
        self.weather_file = os.path.join(self.data_path, "weather_data.csv")
        self.solar_file = os.path.join(self.data_path, "solar_generation_1kW.csv")
        self.weather = Weather(self.weather_file, self.solar_file, hourly_timesteps)
        self.buildings_states_actions_file = buildings_states_actions_file
        self.hourly_timesteps = hourly_timesteps
        self.save_memory = save_memory
        self.building_ids = building_ids
        
        self.net = self.make_grid()

        self.buildings = self.add_houses(6,1)
        self.agents = list(self.buildings.keys())
        self.possible_agents = self.agents[:]
        self.clusters = self.set_clusters()
        self.rl_agents = [x[0] for x in self.clusters]
        self.rl_agents = [j for i in self.rl_agents for j in i]
        self.ncluster = 0

        self.observation_spaces = {k:v.observation_space for k,v in self.buildings.items()}
        self.action_spaces = {k:v.action_space for k,v in self.buildings.items()}

        self.metadata = {'render.modes': [], 'name':"gridlearn"}
        self.ts = 0

        self.voltage_data = []
        self.load_data = []
        self.gen_data = []
        self.reward_data = []
        self.all_rewards =[]

        aspace, ospace = self.get_spaces(self.agents)
        rand_act = {k:v.sample() for k,v in aspace.items()}
        self.step(rand_act)

        self.tester = np.random.uniform(1,5,(5,))

    def normalize_reward(self):
        file = f'models/{self.model_name}/norm_values.json'
        if True:#not os.path.isfile(file):
            timesteps = 4*24*7
            for timestep in range(timesteps):
                action = {k:v.sample() for k,v in self.action_spaces.items()}
                self.step(action)
            all_values = {}
            for k in self.rl_agents:
                self.buildings[k].normalize()
                all_values[k] = self.buildings[k].max_pwr
            with open(file,'w') as f:
                json.dump(all_values, f)
        else:
            for k in self.rl_agents:
                self.buildings[k].normalize(file)
        return self.reset(self.buildings.keys(), True)

    def make_grid(self):
        # make a grid that fits the buildings generated for CityLearn
        net = networks.case33bw()

        # clear the grid of old load values
        load_nodes = net.load['bus']
        res_voltage_nodes = net.bus['name'][net.bus['vn_kv'] == 12.66]
        res_load_nodes = set(load_nodes) & set(res_voltage_nodes)
        net.bus['min_vm_pu'] = 0.7
        net.bus['max_vm_pu'] = 1.3

        for node in res_load_nodes:
            # remove the existing arbitrary load
            net.load.drop(net.load[net.load.bus == node].index, inplace=True)

        conns = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21],
                [25, 26, 27, 28, 29, 30, 31, 32],
                [22, 23, 24]]

        self.pv_buses = [item[-1] for item in conns]
        self.pv_buses += [item[-2] for item in conns]

        mapping = {18:1, 25:5, 22:2}

        net.line.drop(index=net.line[net.line.in_service==False].index, inplace=True)
        net.bus_geodata.at[0,'x'] = 0
        net.bus_geodata.at[0,'y'] = 0
        sw = 'x'
        st = 'y'
        z = -1
        for c in conns:
            z += 1
            for i in range(len(c)):
                if i == 0:
                    if not c[i] == 0:
                        sw = 'y'
                        st = 'x'
                        net.bus_geodata.at[c[i], sw] = net.bus_geodata.at[mapping[c[i]],sw] + 0.2
                        net.bus_geodata.at[c[i], st] = net.bus_geodata.at[mapping[c[i]],st]
                else:
                    net.bus_geodata.at[c[i], sw] = net.bus_geodata.at[c[i-1], sw] + 0.2
                    net.bus_geodata.at[c[i], st] = net.bus_geodata.at[c[i-1], st]

        net.ext_grid.at[0,'vm_pu'] = 1.01

        pp.create_shunt_as_capacitor(net,14,1.2,0)
        pp.create_shunt_as_capacitor(net,24,0.6,0)
        pp.create_shunt_as_capacitor(net,30,1.2,0)
        return net

    def add_houses(self, n, pv_penetration):
        if self.max_num_houses:
            m = 1
        else:
            m = n# + np.random.randint(-2,8)
        houses = []
        b = 0

        # find nodes in the network with residential voltage levels and load infrastructure
        # get the node indexes by their assigned names
        # load_nodes = self.net.load['bus']
        ext_grid_nodes = set(self.net.ext_grid['bus'])
        res_voltage_nodes = set(self.net.bus['name'][self.net.bus['vn_kv'] == 12.66])
        res_load_nodes = res_voltage_nodes - ext_grid_nodes

        buildings = {}
        for existing_node in list(res_load_nodes)[:self.max_num_houses]:
            # remove the existing arbitrary load
            self.net.load.drop(self.net.load[self.net.load.bus == existing_node].index, inplace=True)

            # add n houses at each of these nodes
            for i in range(m):
                # bid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
                if not self.building_ids:
                    with open(self.buildings_states_actions_file) as file:
                        buildings_states_actions = json.load(file)
                    self.building_ids = list(buildings_states_actions.keys())
                    # print(self.building_ids)
                prob = np.ones(len(self.building_ids))
                prob[[1,4,5,6,7,8]] = 10 # building at index 0, 2, 8 correspond to buildings with high energy use
                prob = prob / sum(prob)
                uid = np.random.choice(self.building_ids, p=prob)
                # print(uid)
                bldg = Building(self.data_path, self.climate_zone, self.buildings_states_actions_file, self.hourly_timesteps, uid, self.weather, save_memory=self.save_memory)
                bldg.assign_bus(existing_node)
                bldg.load_index = pp.create_load(self.net, bldg.bus, 0, name=bldg.buildingId) # create a load at the existing bus
                if np.random.uniform() <= 2: # equivalent to 100% PV penetration
                    bldg.gen_index = pp.create_sgen(self.net, bldg.bus, 0, name=bldg.buildingId) # create a generator at the existing bus
                else:
                    bldg.gen_index = -1

                buildings[bldg.buildingId] = bldg
                # bldg.assign_neighbors(self.net)

        from collections import Counter

        types = [v.building_type for v in buildings.values()]
        print(Counter(types))
        return buildings

    def set_clusters(self):
        # determine clusters by geographic order -- an approx for electrical distance
        num_buses = len(self.net.bus)
        num_nodes_per = int(len(self.net.bus)/self.nclusters)
        geo_index = self.net.bus_geodata.sort_values('y').index
        nodal_geo_clusters = [[] for _ in range(self.nclusters)]

        for i in range(self.nclusters):
            nodal_geo_clusters[i] = geo_index.tolist()[i::self.nclusters]

        # make a list of clusters of houses geographically
        house_geo_clusters = []
        for cluster in nodal_geo_clusters:
            filtered = self.net.load.bus.isin(cluster)
            house_geo_clusters += [self.net.load[filtered].name.tolist()]

        # make these into temporal clusters, selecting a portion of agents across each geo_cluster
        clusters = []
        for i in range(self.nclusters):
            cluster = []
            j = i + 0
            for geo_cluster in house_geo_clusters:
                cluster += geo_cluster[j::self.nclusters]
                j = (j + 1) % self.nclusters
            clusters += [cluster]

        # make some of the agents in each cluster RBC agents
        agent_clusters = []
        for cluster in clusters:
            n_agents = int(self.percent_rl * len(cluster))
            rl_agents = np.random.choice(cluster, n_agents).tolist()
            rbc_agents = set(cluster) - set(rl_agents)
            agent_clusters += [(rl_agents, list(rbc_agents))]

        for n in range(self.nclusters):
            for bldg in agent_clusters[n][0]:
                self.buildings[bldg].assign_cluster(n)

        return agent_clusters

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

    def reset(self, agents, reset_logs):
        self.system_losses = []
        self.voltage_dev = []
        return {k:self.buildings[k].reset_timestep(self.net, reset_logs) for k in agents}

    def state(self, agents):
        obs = {k: np.array(self.buildings[k].get_obs(self.net)) for k in agents}
        return obs

    def get_reward(self, agents):
        rewards = {k: self.buildings[k].get_reward(self.net) for k in agents}

        self.reward_data += [sum(rewards.values())]
        # self.all_rewards += [rewards.values()]
        # print(agents)
        return rewards

    def get_done(self, agents):
        dones = {k: (self.buildings[k].time_step >= self.hourly_timesteps*8760) for k in agents}
        return dones

    def get_info(self, agents):
        infos = {agent: {} for agent in agents}
        return infos

    def get_spaces(self, agents):
        #print([self.buildings[k].action_space for k in agents])
        actionspace = {k:self.buildings[k].action_space for k in agents}
        obsspace = {k:self.buildings[k].observation_space for k in agents}
        return actionspace, obsspace

    def step(self, action_dict):
        year_ts = self.ts % (8759*96)
        if year_ts > 90*96 and year_ts < 275*96:
            self.net.shunt.at[0,'q_mvar'] = -1.8
            self.net.shunt.at[1,'q_mvar'] = -0.6
            self.net.shunt.at[2,'q_mvar'] = -1.2
        else:
            self.net.shunt.at[0,'q_mvar'] = -1.2
            self.net.shunt.at[1,'q_mvar'] = -0.01
            self.net.shunt.at[2,'q_mvar'] = -0.01
        for agent in action_dict:
            self.buildings[agent].step(action_dict[agent])

        self.ts += 1
#        if (self.ts % 1000) == 0:
#            print('here', self.ts)
        self.tester = np.random.uniform(1,5,(5,))
        # update the grid based on updated buildings
        self.update_grid()

        # run the grid power flow
        try:
            runpp(self.net, enforce_q_lims=True)
            # print(self.net.res_bus.p_mw.tolist())
        except:
            pp.diagnostic(self.net)
            quit()
            print("QUITTING!!!!")
        rl_agent_keys = list(action_dict.keys())
        rl_agent_keys = [agent for agent in rl_agent_keys if agent in self.rl_agents ]
        obs = self.state(rl_agent_keys)
        self.voltage_data += [list(self.net.res_bus['vm_pu'])]

        self.load_data += [sum(list(self.net.load['p_mw']))]
        self.gen_data += [sum(list(self.net.sgen['p_mw']))]
        return obs, self.get_reward(rl_agent_keys), self.get_done(rl_agent_keys), self.get_info(rl_agent_keys)

    def update_grid(self):
        for agent, bldg in self.buildings.items():
            # Assign the load in MW (from KW in CityLearn)
            self.net.load.at[bldg.load_index, 'p_mw'] = 0.95 * bldg.current_gross_electricity_demand * 0.001
            self.net.load.at[bldg.load_index, 'sn_mva'] = bldg.current_gross_electricity_demand * 0.001

            if bldg.gen_index > -1: # assume PV and battery are both behind the inverter
                self.net.sgen.at[bldg.gen_index, 'p_mw'] = -1 * bldg.current_gross_generation * np.cos(bldg.phi) * 0.001
                self.net.sgen.at[bldg.gen_index, 'q_mvar'] = bldg.current_gross_generation * np.sin(bldg.phi) * 0.001

    def plot_all(self):
        filtered = self.net.load.name.isin(self.rl_agents)
        rl_buses = list(set(self.net.load.loc[filtered].bus))
        fig, ax = plt.subplots(len(rl_buses), figsize=(20, 8*len(rl_buses)))
        x = np.arange(self.ts) / self.hourly_timesteps / 24 / self.nclusters
        # print(len(x))
        for i in range(len(rl_buses)):
            data = np.array(self.voltage_data)[:,rl_buses[i]]
            ax[i].scatter(x, data)
            # print(data)
            ax[i].set_title(f'Bus {rl_buses[i]}')
            ax[i].set_ylabel('Voltage (p.u.)')
            ax[i].set_xlabel('Time (Days)')

        plt.savefig(f'models/{self.model_name}/voltage')
        np.savetxt(f'models/{self.model_name}/voltage.csv', np.array(self.voltage_data), delimiter=",")
        np.savetxt(f'models/{self.model_name}/load.csv', np.array(self.load_data), delimiter=",")
        np.savetxt(f'models/{self.model_name}/reward.csv', np.array(self.reward_data), delimiter=",")
        np.savetxt(f'models/{self.model_name}/solar.csv', np.array(self.gen_data), delimiter=",")
        if not os.path.isdir(f'models/{self.model_name}/homes/'):
            os.mkdir(f'models/{self.model_name}/homes')
        for agent in self.rl_agents[::10]:
            self.buildings[agent].close(self.model_name, write=True)

class MyEnv(ParallelEnv):
    def __init__(self, grid):

        self.agents = grid.clusters[grid.ncluster][0]
        self.rbc_buildings = grid.clusters[grid.ncluster][1]
        self.rbc_agents = None
        grid.ncluster = (grid.ncluster + 1) % grid.nclusters
        self.possible_agents = self.agents[:]
        self.action_spaces, self.observation_spaces = grid.get_spaces(self.agents)

        self.metadata = {'render.modes': [], 'name':"my_env"}
        self.ts = 0

    def set_grid(self, grid):
        self.grid = grid

    def reset(self, reset_logs=True):
        print('calling reset...')
        self.grid.reset(self.agents, reset_logs)
        self.grid.reset([agent.env.buildingId for agent in self.rbc_agents], reset_logs)
        return self.state()

    def state(self):
        return self.grid.state(self.agents)

    def get_reward(self):
        return self.grid.get_reward(self.agents)

    def get_done(self):
        return self.grid.get_done(self.agents)

    def get_info(self):
        return self.grid.get_info(self.agents)

    def initialize_rbc_agents(self, mode='partial'):
        self.rbc_agents = [RBC_Agent(self.grid.buildings[agent]) for agent in self.rbc_buildings]
        for agent in self.rbc_buildings:
            self.grid.buildings[agent].rbc = True

        if mode == 'all':
            self.rbc_agents += [RBC_Agent_v2(self.grid.buildings[agent]) for agent in self.agents]
            for agent in self.agents:
                self.grid.buildings[agent].rbc = True
        return

    def step(self, rl_action_dict):
        action_dict = rl_action_dict

        # get the action_dict for the rbc agents
        if self.rbc_agents:
            for agent in self.rbc_agents:
                #print('calling step on', agent)
                action_dict.update({agent.env.buildingId:agent.predict()})

        # append rbc agent action_dict to the rl agent dict
        return self.grid.step(action_dict)
