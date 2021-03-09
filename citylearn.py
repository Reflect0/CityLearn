import gym
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
import random, string
from gym import spaces
from energy_models import HeatPump, ElectricHeater, EnergyStorage, Building
from reward_function import reward_function_sa, reward_function_ma
from pathlib import Path
from agent import RBC_Agent
gym.logger.set_level(40)

def auto_size(buildings):
    for building in buildings.values():

        # Autosize guarantees that the DHW device is large enough to always satisfy the maximum DHW demand
        if building.dhw_heating_device.nominal_power == 'autosize':

            # If the DHW device is a HeatPump
            if isinstance(building.dhw_heating_device, HeatPump):

                #We assume that the heat pump is always large enough to meet the highest heating or cooling demand of the building
                building.dhw_heating_device.nominal_power = np.array(building.sim_results['dhw_demand']/building.dhw_heating_device.cop_heating).max()

            # If the device is an electric heater
            elif isinstance(building.dhw_heating_device, ElectricHeater):
                building.dhw_heating_device.nominal_power = (np.array(building.sim_results['dhw_demand'])/building.dhw_heating_device.efficiency).max()

        # Autosize guarantees that the cooling device device is large enough to always satisfy the maximum DHW demand
        if building.cooling_device.nominal_power == 'autosize':

            building.cooling_device.nominal_power = (np.array(building.sim_results['cooling_demand'])/ np.repeat(building.cooling_device.cop_cooling, building.hourly_timesteps)).max()

        # Defining the capacity of the storage devices as a number of times the maximum demand
        building.dhw_storage.capacity = max(building.sim_results['dhw_demand'])*building.dhw_storage.capacity
        building.cooling_storage.capacity = max(building.sim_results['cooling_demand'])*building.cooling_storage.capacity

        # Done in order to avoid dividing by 0 if the capacity is 0
        if building.dhw_storage.capacity <= 0.00001:
            building.dhw_storage.capacity = 0.00001
        if building.cooling_storage.capacity <= 0.00001:
            building.cooling_storage.capacity = 0.00001

def subhourly_lin_interp(hourly_data, subhourly_steps):
    """ Returns a linear interpolation of a data array as a list """
    n = len(hourly_data)
    data = np.interp(np.linspace(0, n, n*subhourly_steps), np.arange(n), hourly_data)
    return list(data)

def subhourly_noisy_interp(hourly_data, subhourly_steps):
    """ Returns a noisy distribution of power consumption +/- 5% standard deviation of the original power draw."""
    n = len(hourly_data)
    data = np.repeat(hourly_data, subhourly_steps)
    perturbation = np.random.normal(1.0, 0.05, n*subhourly_steps)
    data = np.multiply(data, perturbation)
    return list(data)

def subhourly_randomdraw_interp(hourly_data, subhourly_steps, dhw_pwr):
    """ Returns a randomized binary distribution where demand = power*time when water is drawn, 0 otherwise.
    Proportion of time with demand at full power corresponds to energy consumption at the hourly interval by E+ """
    data = []
    subhourly_dhw_energy = dhw_pwr / subhourly_steps
    for hour in hourly_data:
        draw_times = np.random.choice(subhourly_steps, int(hour/subhourly_steps), replace=False)
        for i in range(subhourly_steps):
            if i in draw_times:
                data += [subhourly_dhw_energy]
            else:
                data += [0]
    return list(data)

def set_dhw_draws(buildings):
    for uid, building in buildings.items():
        building.sim_results['dhw_demand'] = subhourly_randomdraw_interp(building.sim_results['dhw_demand'], building.hourly_timesteps, building.dhw_heating_device.nominal_power)

def building_loader(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions, n_buildings, hourly_timesteps, save_memory = True):
    with open(building_attributes) as json_file:
        data = json.load(json_file)

    data = {k:v for k,v in data.items() if k in building_ids}

    buildings, observation_spaces, action_spaces = {},[],{}
    s_low_central_agent, s_high_central_agent, appended_states = [], [], []
    a_low_central_agent, a_high_central_agent, appended_actions = [], [], []
    all_data = list(zip(data, data.values()))
    for _ in range(n_buildings):
        uid, attributes = random.choice(all_data) # @akp, iterate through buildings randomly to create duplicates of building types
        if uid in building_ids:
            heat_pump = HeatPump(nominal_power = attributes['Heat_Pump']['nominal_power'],
                                 eta_tech = attributes['Heat_Pump']['technical_efficiency'],
                                 t_target_heating = attributes['Heat_Pump']['t_target_heating'],
                                 t_target_cooling = attributes['Heat_Pump']['t_target_cooling'], save_memory = save_memory)

            electric_heater = ElectricHeater(nominal_power = attributes['Electric_Water_Heater']['nominal_power'],
                                             efficiency = attributes['Electric_Water_Heater']['efficiency'], save_memory = save_memory)

            chilled_water_tank = EnergyStorage(capacity = attributes['Chilled_Water_Tank']['capacity'],
                                               loss_coeff = attributes['Chilled_Water_Tank']['loss_coefficient'], save_memory = save_memory)

            dhw_tank = EnergyStorage(capacity = attributes['DHW_Tank']['capacity'],
                                     loss_coeff = attributes['DHW_Tank']['loss_coefficient'], save_memory = save_memory)

            building = Building(buildingId = uid, hourly_timesteps=hourly_timesteps, dhw_storage = dhw_tank, cooling_storage = chilled_water_tank, dhw_heating_device = electric_heater, cooling_device = heat_pump, save_memory = save_memory)

            data_file = str(uid) + '.csv'
            simulation_data = data_path / data_file
            with open(simulation_data) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['cooling_demand'] = subhourly_lin_interp(data['Cooling Load [kWh]'], hourly_timesteps)
            building.sim_results['dhw_demand'] = list(data['DHW Heating [kWh]'])
            building.sim_results['non_shiftable_load'] = subhourly_noisy_interp(data['Equipment Electric Power [kWh]'], hourly_timesteps)
            building.sim_results['month'] = list(np.repeat(data['Month'], hourly_timesteps))
            building.sim_results['day'] = list(np.repeat(data['Day Type'], hourly_timesteps))
            building.sim_results['hour'] = list(np.repeat(data['Hour'], hourly_timesteps))
            building.sim_results['daylight_savings_status'] = list(np.repeat(data['Daylight Savings Status'], hourly_timesteps))
            building.sim_results['t_in'] = subhourly_lin_interp(data['Indoor Temperature [C]'], hourly_timesteps)
            building.sim_results['avg_unmet_setpoint'] = subhourly_lin_interp(data['Average Unmet Cooling Setpoint Difference [C]'], hourly_timesteps)
            building.sim_results['rh_in'] = subhourly_lin_interp(data['Indoor Relative Humidity [%]'], hourly_timesteps)

            with open(weather_file) as csv_file:
                weather_data = pd.read_csv(csv_file)

            building.sim_results['t_out'] = subhourly_lin_interp(weather_data['Outdoor Drybulb Temperature [C]'], hourly_timesteps)
            building.sim_results['rh_out'] = subhourly_lin_interp(weather_data['Outdoor Relative Humidity [%]'], hourly_timesteps)
            building.sim_results['diffuse_solar_rad'] = subhourly_lin_interp(weather_data['Diffuse Solar Radiation [W/m2]'], hourly_timesteps)
            building.sim_results['direct_solar_rad'] = subhourly_lin_interp(weather_data['Direct Solar Radiation [W/m2]'], hourly_timesteps)

            # Reading weather forecasts
            building.sim_results['t_out_pred_6h'] = subhourly_lin_interp(weather_data['6h Prediction Outdoor Drybulb Temperature [C]'], hourly_timesteps)
            building.sim_results['t_out_pred_12h'] = subhourly_lin_interp(weather_data['12h Prediction Outdoor Drybulb Temperature [C]'], hourly_timesteps)
            building.sim_results['t_out_pred_24h'] = subhourly_lin_interp(weather_data['24h Prediction Outdoor Drybulb Temperature [C]'], hourly_timesteps)

            building.sim_results['rh_out_pred_6h'] = subhourly_lin_interp(weather_data['6h Prediction Outdoor Relative Humidity [%]'], hourly_timesteps)
            building.sim_results['rh_out_pred_12h'] = subhourly_lin_interp(weather_data['12h Prediction Outdoor Relative Humidity [%]'], hourly_timesteps)
            building.sim_results['rh_out_pred_24h'] = subhourly_lin_interp(weather_data['24h Prediction Outdoor Relative Humidity [%]'], hourly_timesteps)

            building.sim_results['diffuse_solar_rad_pred_6h'] = subhourly_lin_interp(weather_data['6h Prediction Diffuse Solar Radiation [W/m2]'], hourly_timesteps)
            building.sim_results['diffuse_solar_rad_pred_12h'] = subhourly_lin_interp(weather_data['12h Prediction Diffuse Solar Radiation [W/m2]'], hourly_timesteps)
            building.sim_results['diffuse_solar_rad_pred_24h'] = subhourly_lin_interp(weather_data['24h Prediction Diffuse Solar Radiation [W/m2]'], hourly_timesteps)

            building.sim_results['direct_solar_rad_pred_6h'] = subhourly_lin_interp(weather_data['6h Prediction Direct Solar Radiation [W/m2]'], hourly_timesteps)
            building.sim_results['direct_solar_rad_pred_12h'] = subhourly_lin_interp(weather_data['12h Prediction Direct Solar Radiation [W/m2]'], hourly_timesteps)
            building.sim_results['direct_solar_rad_pred_24h'] = subhourly_lin_interp(weather_data['24h Prediction Direct Solar Radiation [W/m2]'], hourly_timesteps)

            # Reading the building attributes
            building.building_type = attributes['Building_Type']
            building.climate_zone = attributes['Climate_Zone']
            building.solar_power_capacity = attributes['Solar_Power_Installed(kW)']

            with open(solar_profile) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['solar_gen'] = subhourly_lin_interp(attributes['Solar_Power_Installed(kW)']*data['Hourly Data: AC inverter power (W)']/1000, hourly_timesteps)

            # Finding the max and min possible values of all the states, which can then be used by the RL agent to scale the states and train any function approximators more effectively
            s_low, s_high = [], []
            for state_name, value in zip(buildings_states_actions[uid]['states'], buildings_states_actions[uid]['states'].values()):
                if value == True:
                    if state_name == "net_electricity_consumption":
                        # lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate. Scaling this state-variable using these bounds may result in normalized values above 1 or below 0.
                        _net_elec_cons_upper_bound = max(np.array(building.sim_results['non_shiftable_load']) - np.array(building.sim_results['solar_gen']) + np.array(np.repeat(building.sim_results['dhw_demand'], hourly_timesteps))/.8 + np.array(building.sim_results['cooling_demand']) + building.dhw_storage.capacity/.8 + building.cooling_storage.capacity/2)
                        s_low.append(0.)
                        s_high.append(_net_elec_cons_upper_bound)
                        s_low_central_agent.append(0.)
                        s_high_central_agent.append(_net_elec_cons_upper_bound)

                    elif state_name == "relative_voltage":
                        # @akp, added relative voltage to give homes their voltage ranked against the community max/min
                        s_low.append(0.) # the house is the lowest voltage in the community
                        s_high.append(1.)

                    elif state_name == "total_voltage_spread":
                        # @akp, added total voltage spread to give a sense of the total loss incurred by the community. without the total voltage spread state "relative_voltage" is more or less meaningless. (total_voltage_spread = how much the community is penaltized, relative_voltage = what that house can do to fix the issue)
                        s_low.append(0.) # @akp?, not sure what the typical spread in v pu would be.
                        s_high.append(1.)

                        # @akp?, still not sure how the central agent s.append() works...
                        s_low_central_agent.append(0.)
                        s_high_central_agent.append(1.)

                    elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                        s_low.append(min(building.sim_results[state_name]))
                        s_high.append(max(building.sim_results[state_name]))

                        # Create boundaries of the observation space of a centralized agent (if a central agent is being used instead of decentralized ones). We include all the weather variables used as states, and use the list appended_states to make sure we don't include any repeated states (i.e. weather variables measured by different buildings)
                        if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                            s_low_central_agent.append(min(building.sim_results[state_name]))
                            s_high_central_agent.append(max(building.sim_results[state_name]))

                        elif state_name not in appended_states:
                            s_low_central_agent.append(min(building.sim_results[state_name]))
                            s_high_central_agent.append(max(building.sim_results[state_name]))
                            appended_states.append(state_name)

                    else:
                        s_low.append(0.0)
                        s_high.append(1.0)
                        s_low_central_agent.append(0.0)
                        s_high_central_agent.append(1.0)

            '''The energy storage (tank) capacity indicates how many times bigger the tank is compared to the maximum hourly energy demand of the building (cooling or DHW respectively), which sets a lower bound for the action of 1/tank_capacity, as the energy storage device can't provide the building with more energy than it will ever need for a given hour. The heat pump is sized using approximately the maximum hourly energy demand of the building (after accounting for the COP, see function autosize). Therefore, we make the fair assumption that the action also has an upper bound equal to 1/tank_capacity. This boundaries should speed up the learning process of the agents and make them more stable rather than if we just set them to -1 and 1. I.e. if Chilled_Water_Tank.Capacity is 3 (3 times the max. hourly demand of the building in the entire year), its actions will be bounded between -1/3 and 1/3'''
            a_low, a_high = [], []
            for action_name, value in zip(buildings_states_actions[uid]['actions'], buildings_states_actions[uid]['actions'].values()):
                if value == True:
                    if action_name =='cooling_storage':

                        # Avoid division by 0
                        if attributes['Chilled_Water_Tank']['capacity'] > 0.000001:
                            a_low.append(max(-1.0/attributes['Chilled_Water_Tank']['capacity'], -1.0))
                            a_high.append(min(1.0/attributes['Chilled_Water_Tank']['capacity'], 1.0))
                            a_low_central_agent.append(max(-1.0/attributes['Chilled_Water_Tank']['capacity'], -1.0))
                            a_high_central_agent.append(min(1.0/attributes['Chilled_Water_Tank']['capacity'], 1.0))
                        else:
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            a_low_central_agent.append(-1.0)
                            a_high_central_agent.append(1.0)

                    elif action_name == 'dhw_storage':
                        if attributes['DHW_Tank']['capacity'] > 0.000001:
                            a_low.append(max(-1.0/attributes['DHW_Tank']['capacity'], -1.0))
                            a_high.append(min(1.0/attributes['DHW_Tank']['capacity'], 1.0))
                            a_low_central_agent.append(max(-1.0/attributes['DHW_Tank']['capacity'], -1.0))
                            a_high_central_agent.append(min(1.0/attributes['DHW_Tank']['capacity'], 1.0))
                        else:
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            a_low_central_agent.append(-1.0)
                            a_high_central_agent.append(1.0)

                    elif action_name == 'pv_curtail':
                        # pv curtailment of apparent power, S
                        a_low.append(-1.0)
                        a_high.append(1.0)
                        a_low_central_agent.append(-1.0)
                        a_high_central_agent.append(1.0)

                    elif action_name == 'pv_vm':
                        # smart inverter voltage control @constance?
                        a_low.append(-1.0)
                        a_high.append(1.0)
                        a_low_central_agent.append(-1.0)
                        a_high_central_agent.append(1.0)


            building.set_state_space(np.array(s_high), np.array(s_low))
            building.set_action_space(np.array(a_high), np.array(a_low))

            unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            buildings[unique_id] = building

            observation_spaces.append(building.observation_space)
            action_spaces[unique_id] = building.action_space

    observation_space_central_agent = spaces.Box(low=np.float32(np.array(s_low_central_agent)), high=np.float32(np.array(s_high_central_agent)), dtype=np.float32)
    action_space_central_agent = spaces.Box(low=np.float32(np.array(a_low_central_agent)), high=np.float32(np.array(a_high_central_agent)), dtype=np.float32)

    for building in buildings.values():

        # If the DHW device is a HeatPump
        if isinstance(building.dhw_heating_device, HeatPump):

            # Calculating COPs of the heat pumps for every hour
            building.dhw_heating_device.cop_heating = building.dhw_heating_device.eta_tech*(building.dhw_heating_device.t_target_heating + 273.15)/(building.dhw_heating_device.t_target_heating - weather_data['Outdoor Drybulb Temperature [C]'])
            building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating < 0] = 20.0
            building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating > 20] = 20.0
            building.dhw_heating_device.cop_heating = building.dhw_heating_device.cop_heating.to_numpy()

        building.cooling_device.cop_cooling = building.cooling_device.eta_tech*(building.cooling_device.t_target_cooling + 273.15)/(weather_data['Outdoor Drybulb Temperature [C]'] - building.cooling_device.t_target_cooling)
        building.cooling_device.cop_cooling[building.cooling_device.cop_cooling < 0] = 20.0
        building.cooling_device.cop_cooling[building.cooling_device.cop_cooling > 20] = 20.0
        building.cooling_device.cop_cooling = building.cooling_device.cop_cooling.to_numpy()

        building.reset()

    auto_size(buildings)

    set_dhw_draws(buildings) # @akp, kinda janky but until this point the dhw nominal power isn't set

    return buildings, observation_spaces, action_spaces, observation_space_central_agent, action_space_central_agent

class CityLearn(gym.Env):
    def __init__(self, data_path, building_attributes, weather_file, solar_profile, building_ids, hourly_timesteps, buildings_states_actions = None, simulation_period = (0,8759), cost_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption'], central_agent = False, verbose = 0, n_buildings=None):

        np.random.seed(12)
        random.seed(12)

        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)

        self.buildings_states_actions_filename = buildings_states_actions
        self.buildings_net_electricity_demand = []
        self.building_attributes = building_attributes
        self.solar_profile = solar_profile
        self.building_ids = building_ids
        self.cost_function = cost_function
        self.cost_rbc = None
        self.data_path = data_path
        self.weather_file = weather_file
        self.central_agent = central_agent
        self.loss = []
        self.verbose = verbose
        self.hourly_timesteps = hourly_timesteps

        self.simulation_period = simulation_period
        self.uid = None
        if not n_buildings: # added as a parameter @AKP
            self.n_buildings = len(building_ids)
        else:
            self.n_buildings = n_buildings

        self.buildings, self.observation_spaces, self.action_spaces, self.observation_space, self.action_space = building_loader(data_path, building_attributes, weather_file, solar_profile, building_ids, self.buildings_states_actions, self.n_buildings, self.hourly_timesteps)

        self.buildings_states_actions = {k:self.buildings_states_actions[self.buildings[k].buildingId] for k in self.buildings}

        self.reset()

    def get_state_action_spaces(self):
        return self.observation_spaces, self.action_spaces

    def next_hour(self):
        self.time_step = next(self.hour)
        for building in self.buildings.values():
            building.time_step = self.time_step

    def get_building_information(self):

        np.seterr(divide='ignore', invalid='ignore')
        # Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand
        building_info = {}
        for uid, building in self.buildings.items():
            building_info[uid] = {}
            building_info[uid]['building_type'] = building.building_type
            building_info[uid]['climate_zone'] = building.climate_zone
            building_info[uid]['solar_power_capacity (kW)'] = round(building.solar_power_capacity, 3)
            building_info[uid]['Annual_DHW_demand (kWh)'] = round(sum(building.sim_results['dhw_demand']), 3)
            building_info[uid]['Annual_cooling_demand (kWh)'] = round(sum(building.sim_results['cooling_demand']), 3)
            building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] = round(sum(building.sim_results['non_shiftable_load']), 3)

            building_info[uid]['Correlations_DHW'] = {}
            building_info[uid]['Correlations_cooling_demand'] = {}
            building_info[uid]['Correlations_non_shiftable_load'] = {}

            for uid_corr, building_corr in self.buildings.items():
                if uid_corr != uid:
                    building_info[uid]['Correlations_DHW'][uid_corr] = round((np.corrcoef(np.array(building.sim_results['dhw_demand']), np.array(building_corr.sim_results['dhw_demand'])))[0][1], 3)
                    building_info[uid]['Correlations_cooling_demand'][uid_corr] = round((np.corrcoef(np.array(building.sim_results['cooling_demand']), np.array(building_corr.sim_results['cooling_demand'])))[0][1], 3)
                    building_info[uid]['Correlations_non_shiftable_load'][uid_corr] = round((np.corrcoef(np.array(building.sim_results['non_shiftable_load']), np.array(building_corr.sim_results['non_shiftable_load'])))[0][1], 3)

        return building_info

    def step(self, actions):

        self.buildings_net_electricity_demand = []
        electric_demand = 0
        elec_consumption_dhw_storage = 0
        elec_consumption_cooling_storage = 0
        elec_consumption_dhw_total = 0
        elec_consumption_cooling_total = 0
        elec_consumption_appliances = 0
        elec_generation = 0

        if self.central_agent:
            # If the agent is centralized, all the actions for all the buildings are provided as an ordered list of numbers. The order corresponds to the order of the buildings as they appear on the file building_attributes.json, and only considering the buildings selected for the simulation by the user (building_ids).
            for uid, building in self.buildings.items():

                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    # Cooling
                    _electric_demand_cooling = building.set_storage_cooling(actions[0])
                    actions = actions[1:]
                    elec_consumption_cooling_storage += building._electric_consumption_cooling_storage
                else:
                    _electric_demand_cooling = 0

                if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                    # DHW
                    _electric_demand_dhw = building.set_storage_heating(actions[0])
                    actions = actions[1:]
                    elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                else:
                    _electric_demand_dhw = 0

                if self.buildings_states_actions[uid]['actions']['pv_curtail']:
                    # solar power curtailment
                    _solar_generation = building.get_solar_power(actions[0])
                    actions = actions[1:]
                else:
                    _solar_generation = building.get_solar_power()
                elec_generation += _solar_generation # because the default is to not curtail pv production.

                if self.buildings_states_actions[uid]['actions']['pv_vm']:
                    # @constance look at this action for changing the smart inverter actions to P,Q
                    building.set_target_vm(actions[0])
                    actions = actions[1:]
                else:
                    building.set_target_vm()

                # Total heating and cooling electrical loads
                elec_consumption_cooling_total += _electric_demand_cooling
                elec_consumption_dhw_total += _electric_demand_dhw

                # Electrical appliances
                _non_shiftable_load = building.get_non_shiftable_load()
                elec_consumption_appliances += _non_shiftable_load

                # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
                building_electric_demand = round(_electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation, 4)

                # Electricity consumed by every building
                building.current_net_electricity_demand = building_electric_demand
                self.buildings_net_electricity_demand.append(-building_electric_demand) # >0 if solar generation > electricity consumption

                # Total electricity consumption
                electric_demand += building_electric_demand

            assert len(actions) == 0, 'Some of the actions provided were not used'

        else:

            # assert len(actions) == self.n_buildings, "The length of the list of actions should match the length of the list of buildings."
            # for a, (uid, building) in zip(actions, self.buildings.items()):
            for uid, a in actions.items():
                building = self.buildings[uid]
                # assert sum(self.buildings_states_actions[uid]['actions'].values()) == len(a), "The number of input actions for building "+str(uid)+" must match the number of actions defined in the list of building attributes." + str(self.buildings_states_actions[uid]['actions'].values()) + str(a)
                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    # Cooling will always be the first action available
                    # print("setting cooling storage...")
                    _electric_demand_cooling = building.set_storage_cooling(a[0])
                    elec_consumption_cooling_storage += building._electric_consumption_cooling_storage
                    a = a[1:]
                else:
                    _electric_demand_cooling = 0

                if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                    # DHW
                    # print("setting dhw storage...")
                    _electric_demand_dhw = building.set_storage_heating(a[0])
                    elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                    a = a[1:]
                else:
                    _electric_demand_dhw = 0

                if self.buildings_states_actions[uid]['actions']['pv_curtail']:
                    # Solar power
                    # print("setting pv curtailment...")
                    _solar_generation = building.get_solar_power(a[0])
                    elec_generation += _solar_generation
                    a = a[1:]
                else:
                    _solar_generation = building.get_solar_power()
                    elec_generation += _solar_generation

                if self.buildings_states_actions[uid]['actions']['pv_vm']:
                    # print("setting pv voltage")
                    building.set_target_vm(a[0])
                    a = a[1:]
                else:
                    building.set_target_vm()

                # Total heating and cooling electrical loads
                elec_consumption_cooling_total += _electric_demand_cooling
                elec_consumption_dhw_total += _electric_demand_dhw

                # Electrical appliances
                _non_shiftable_load = building.get_non_shiftable_load()
                elec_consumption_appliances += _non_shiftable_load

                # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
                building_electric_demand = round(_electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation, 4)

                # Electricity consumed by every building
                building.current_net_electricity_demand = building_electric_demand
                self.buildings_net_electricity_demand.append(-building_electric_demand)

                # Total electricity consumption
                electric_demand += building_electric_demand

        self.aux_grid_func()
        self.next_hour()

        if self.central_agent:
            s, s_appended = [], []
            for uid, building in self.buildings.items():

                # If the agent is centralized, we append the states avoiding repetition. I.e. if multiple buildings share the outdoor temperature as a state, we only append it once to the states of the central agent. The variable s_appended is used for this purpose.
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name not in s_appended:
                            if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                                s.append(building.sim_results[state_name][self.time_step])
                            elif state_name == 'net_electricity_consumption':
                                s.append(building.current_net_electricity_demand)
                            elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                                s.append(building.sim_results[state_name][self.time_step])
                                s_appended.append(state_name)
                            elif state_name == 'cooling_storage_soc':
                                s.append(building.cooling_storage._soc/building.cooling_storage.capacity)
                            elif state_name == 'dhw_storage_soc':
                                s.append(building.dhw_storage._soc/building.dhw_storage.capacity)
            self.state = np.array(s)
            rewards = reward_function_sa(self.buildings_net_electricity_demand)
            self.cumulated_reward_episode += rewards

        else:
            # If the controllers are decentralized, we append all the states to each associated agent's list of states.
            self.state = []
            # for uid, building in self.buildings.items():
            for uid in actions.keys():
                building = self.buildings[uid]
#             for k, building in self.buildings.items():
#                 uid = building.buildingId
                s = []
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name == 'net_electricity_consumption':
                            s.append(building.current_net_electricity_demand)

                        elif state_name == 'relative_voltage':
                            if self.time_step <= 1:
                                s.append(0.5)
                            else:
                                ranked_voltage = self.net.res_bus['vm_pu'].rank(pct=True)[self.net.load.loc[self.net.load['name']==uid].bus]
                                s.append(ranked_voltage)
                        elif state_name == 'total_voltage_spread':
                            if self.time_step <= 1:
                                s.append(0)
                            else:
                                voltage_spread = 0
                                for index, line in self.net.line.iterrows():
                                    voltage_spread += abs(self.net.res_bus.loc[line.to_bus].vm_pu - self.net.res_bus.loc[line.from_bus].vm_pu)
                                s.append(voltage_spread)

                        elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                            s.append(building.sim_results[state_name][self.time_step])
                        elif state_name == 'cooling_storage_soc':
                            s.append(building.cooling_storage._soc/building.cooling_storage.capacity)
                        elif state_name == 'dhw_storage_soc':
                            s.append(building.dhw_storage._soc/building.dhw_storage.capacity)

                self.state.append(np.array(s))
            self.state = np.array(self.state, dtype=object)
            sys_losses = self.system_losses[-1] if self.time_step > 1 else 0
            rewards = self.reward_function.get_rewards(sys_losses)
            self.cumulated_reward_episode += sum(rewards)

        # Control variables which are used to display the results and the behavior of the buildings at the district level.
        self.net_electric_consumption.append(np.float32(electric_demand))
        self.electric_consumption_dhw_storage.append(np.float32(elec_consumption_dhw_storage))
        self.electric_consumption_cooling_storage.append(np.float32(elec_consumption_cooling_storage))
        self.electric_consumption_dhw.append(np.float32(elec_consumption_dhw_total))
        self.electric_consumption_cooling.append(np.float32(elec_consumption_cooling_total))
        self.electric_consumption_appliances.append(np.float32(elec_consumption_appliances))
        self.electric_generation.append(np.float32(elec_generation))
        self.net_electric_consumption_no_storage.append(np.float32(electric_demand-elec_consumption_cooling_storage-elec_consumption_dhw_storage))
        self.net_electric_consumption_no_pv_no_storage.append(np.float32(electric_demand + elec_generation - elec_consumption_cooling_storage - elec_consumption_dhw_storage))

        terminal = self._terminal()
        return (self._get_ob(), rewards, terminal, {})

    def get_cvr_electricity_demand(self):
        return self.buildings_net_electricity_demand

    def aux_grid_func(self):
        return("Auxiliary grid function has not been implemented yet.")

    def reset_baseline_cost(self):
        self.cost_rbc = None

    def reset(self):

        #Initialization of variables
        self.hour = iter(np.array(range(self.simulation_period[0], self.simulation_period[1] + 1)))
        self.next_hour()

        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        self.electric_consumption_dhw_storage = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_cooling = []
        self.electric_consumption_appliances = []
        self.electric_generation = []

        self.cumulated_reward_episode = 0

        if self.central_agent:
            s, s_appended = [], []
            for uid, building in self.buildings.items():
#                 uid = "Building_"+str(building.building_type)
                building.reset()
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if state_name not in s_appended:
                        if value == True:
                            if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                                s.append(building.sim_results[state_name][self.time_step])
                            elif state_name == 'net_electricity_consumption':
                                s.append(building.current_net_electricity_demand)
                            elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                                s.append(building.sim_results[state_name][self.time_step])
                                s_appended.append(state_name)
                            elif state_name == 'cooling_storage_soc':
                                s.append(0.0)
                            elif state_name == 'dhw_storage_soc':
                                s.append(0.0)
            self.state = np.array(s)
        else:
            self.reward_function = reward_function_ma(len(self.building_ids), self.get_building_information())

            self.state = []
            for uid, building in self.buildings.items():
#                 uid = "Building_" + str(building.building_type)
                building.reset()
                s = []
                for state_name, value in zip(self.buildings_states_actions[uid]['states'], self.buildings_states_actions[uid]['states'].values()):
                    if value == True:
                        if state_name == 'net_electricity_consumption':
                            s.append(building.current_net_electricity_demand)
                        elif state_name == 'total_voltage_spread':
                            if self.time_step == 0:
                                s.append(0)
                            else:
                                voltage_spread = s.append(max(self.net.res_bus.vm_pu)-min(self.net.res_bus.vm_pu))
                        elif state_name == 'relative_voltage':
                            if self.time_step == 0:
                                s.append(0.5)
                            else:
                                house_voltage = self.net.res_bus[self.net.res_bus['name'] == uid]
                                s.append(house_voltage / spread)
                        elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc':
                            s.append(building.sim_results[state_name][self.time_step])
                        elif state_name == 'cooling_storage_soc':
                            s.append(0.0)
                        elif state_name == 'dhw_storage_soc':
                            s.append(0.0)

                self.state.append(np.array(s, dtype=np.float32))

            self.state = np.array(self.state)

        return self._get_ob()

    def _get_ob(self):
        return self.state

    def _terminal(self):
        is_terminal = bool(self.time_step >= self.simulation_period[1])
        if is_terminal:
            for building in self.buildings.values():
                building.terminate()

            # When the simulation is over, convert all the control variables to numpy arrays so they are easier to plot.
            self.net_electric_consumption = np.array(self.net_electric_consumption)
            self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)
            self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
            self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
            self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
            self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
            self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
            self.electric_generation = np.array(self.electric_generation)
            self.loss.append([i for i in self.get_baseline_cost().values()])

            if self.verbose == 1:
                print('Cumulated reward: '+str(self.cumulated_reward_episode))

        return is_terminal

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_buildings_net_electric_demand(self):
        return self.buildings_net_electricity_demand

    def get_rbc_cost(self):
        # Running the reference rule-based controller to find the baseline cost
        if self.cost_rbc is None:
            env_rbc = CityLearn(self.data_path, self.building_attributes, self.weather_file, self.solar_profile, self.building_ids, hourly_timesteps=self.hourly_timesteps, buildings_states_actions = self.buildings_states_actions_filename, simulation_period = self.simulation_period, cost_function = self.cost_function, central_agent = False, n_buildings=self.n_buildings)
            _, actions_spaces = env_rbc.get_state_action_spaces()

            #Instantiatiing the control agent(s)
            agent_rbc = RBC_Agent(self.buildings_states_actions)

            state = env_rbc.reset()
            done = False
            while not done:
                action = agent_rbc.select_action([list(env_rbc.buildings.values())[0].sim_results['hour'][env_rbc.time_step]])
                next_state, rewards, done, _ = env_rbc.step(action)
                state = next_state
            self.cost_rbc = env_rbc.get_baseline_cost()

    def cost(self):

        self.get_rbc_cost()

        # Compute the costs normalized by the baseline costs
        cost = {}
        self.net_electric_consumption = np.array(self.net_electric_consumption)
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs((self.net_electric_consumption - np.roll(self.net_electric_consumption,1))[1:]).sum()/self.cost_rbc['ramping']

        # Finds the load factor for every month (average monthly demand divided by its maximum peak), and averages all the load factors across the 12 months. The metric is one minus the load factor.
        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1-np.mean(self.net_electric_consumption[i:i+int(8760/12)])/ np.max(self.net_electric_consumption[i:i+int(8760/12)]) for i in range(0,len(self.net_electric_consumption), int(8760/12))])/self.cost_rbc['1-load_factor']

        # Average of all the daily peaks of the 365 day of the year. The peaks are calculated using the net energy demand of the whole district of buildings.
        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([np.max(self.net_electric_consumption[i:i+24]) for i in range(0,len(self.net_electric_consumption),24)])/self.cost_rbc['average_daily_peak']

        # Peak demand of the district for the whole year period.
        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = np.max(self.net_electric_consumption)/self.cost_rbc['peak_demand']

        # Positive net electricity consumption for the whole district. It is clipped at a min. value of 0 because the objective is to minimize the energy consumed in the district, not to profit from the excess generation. (Island operation is therefore incentivized)
        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum()/self.cost_rbc['net_electricity_consumption']

        # Total line losses of the network
        if 'system_losses' in self.cost_function:
            cost['system_losses'] = -1*np.sum(self.system_losses)/self.cost_rbc['system_losses']

        # Not used for the challenge
        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0)**2).sum()/self.cost_rbc['quadratic']

        cost['total'] = np.mean([c for c in cost.values()])

        return cost

    def get_baseline_cost(self):

        # Computes the costs for the Rule-based controller, which are used to normalized the actual costs.
        cost = {}
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs((self.net_electric_consumption - np.roll(self.net_electric_consumption,1))[1:]).sum()

        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1 - np.mean(self.net_electric_consumption[i:i+int(8760/12)])/ np.max(self.net_electric_consumption[i:i+int(8760/12)]) for i in range(0, len(self.net_electric_consumption), int(8760/12))])

        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([np.max(self.net_electric_consumption[i:i+24]) for i in range(0, len(self.net_electric_consumption), 24)])

        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = np.max(self.net_electric_consumption)

        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = np.array(self.net_electric_consumption).clip(min=0).sum()

        if 'system_losses' in self.cost_function:
            cost['system_losses'] = -1*np.sum(self.system_losses)

        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (np.array(self.net_electric_consumption).clip(min=0)**2).sum()

        return cost
