import pandapower as pp
from pandapower import runpp
from pandapower.plotting import simple_plotly, pf_res_plotly
import pandapower.networks as networks
from citylearn import CityLearn
from citylearn import RBC_Agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

class GridLearn: # not a super class of the CityLearn environment
    def __init__(self):
        self.name = "test"
        self.net = self.make_grid()
        self.next_load_bus = 1

    def make_test_grid(self):
        net = pp.create_empty_network(name="single bus network")

        ex_grid = pp.create_bus(net, name=0, vn_kv=12.66, geodata=(0,0))
        pp.create_ext_grid(net, ex_grid, vm_pu=1.02, va_degree=50)

        bus1 = pp.create_bus(net, name=1, vn_kv=12.66, geodata=(0,1))
        load1 = pp.create_load(net, 1, p_mw=0)

        main_line = pp.create_line(net, ex_grid, bus1, 0.5, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV") # arbitrary line

        return net

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
        load_nodes = self.net.load['bus']
        res_voltage_nodes = self.net.bus['name'][self.net.bus['vn_kv'] == 12.66]
        res_load_nodes = set(load_nodes) & set(res_voltage_nodes)

        # add a residential distribution feeder type to the PandaPower network
        # Assume for now ~250A per home on "94-AL1/15-ST1A 0.4" lines rated @ 350A

        # for geometric placement of nodes
        delta_x = 0.2
        delta_y = 0.2

        all_buildings = list(self.buildings.keys())

        for existing_node in res_load_nodes:
            # remove the existing arbitrary load
            self.net.load.drop(self.net.load[self.net.load.bus == existing_node].index, inplace=True)

            # get geodata of this load
            existing_x = self.net.bus_geodata['x'][existing_node]
            existing_y = self.net.bus_geodata['y'][existing_node]

            # add n houses at each of these nodes
            for i in range(n):
                # bid = all_buildings[b] # get a building in the order they were initialized
                # b += 1
                # new_x = existing_x + np.cos(2 * np.pi/n * i) * delta_x
                # new_y = existing_y + np.sin(2 * np.pi/n * i) * delta_y
                # new_house = pp.create_bus(self.net, name=bid, vn_kv=12.66, max_vm_pu=1.2, min_vm_pu=0.8, zone=1, geodata=(new_x, new_y))
                # new_feeder = pp.create_line(self.net, new_house, existing_node, 0.5, "94-AL1/15-ST1A 0.4", max_loading_percent=100)
                # new_house_load = pp.create_load(self.net, new_house, 0, name=bid)
                new_house_load = pp.create_load(self.net, existing_node, 0, name=bid) # create a load at the existing bus
#                 if self.buildings_states_actions[bid]['pv_curtail']:
                if np.random.uniform() <= pv_penetration:
                    new_house_pv = pp.create_sgen(self.net, existing_node, 0, name=bid) # create a generator at the existing bus
                # houses += [new_house]
        return

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
        return super().reset()

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
