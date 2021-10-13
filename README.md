# GridLearn
GridLearn is a spin off of the open source OpenAI Gym environment CityLearn. Like CityLearn, GridLearn is a testbed for the implementation of Multi-Agent Reinforcement Learning (MARL) in building energy coordination and demand response in cities. GridLearn builds off of CityLearn by implementing power flow simulations in addition to aggregating the district's demand profiles. The current release of GridLearn implements all actions in parallel (Fig 1) -- future releases will consider alternate action selection (Fig 2).

<img width="400" alt="single_cluster" src="https://github.com/apigott/CityLearn/blob/master/images/single_cluster.png">
<img width="400" alt="multiple_clusters" src="https://github.com/apigott/CityLearn/blob/master/images/multiple_clusters.png">

<!-- ![single-cluser](https://github.com/apigott/CityLearn/blob/master/images/single_cluster.png) -->
<!-- ![multiple-cluser](https://github.com/apigott/CityLearn/blob/master/images/multiple_clusters.png) -->

## Description
Due to aging infrastructure and increasing penetration of PV, distribution networks are now facing voltage issues (both under- and overvoltages). To overcome voltage regulation issues grid operators have historically used capacitor banks to increase the voltage at different points of the feeder. However, as PV penetration rates rise and overvoltages become more prevelant this is not a viable solution. To that end many studies have been done on the efficacy of smart inverters and demand response to help perform voltage regulation.
## Requirements
CityLearn requires the installation of the following Python libraries:
- Pandas 0.24.2 or older
- Pandapower 2.0
- Numpy 1.16.4 or older
- Gym 0.14.0
- Json 2.0.9

In order to run the main files with the sample agent provided you will need:
- stable-baselines3 3.0

GridLearn may still work with some earlier versions of these libraries, but we have tested it with those.

## Unit Test Files
- [For training a multiple RL agents simultaneously](/tests/pezTest.py)
- [For evaluating a trained MARL scenario](/tests/pezEval.py)
- [For evaluating a baseline scenario](/tests/baselineEval.py)
- [For inspecting and unit testing class instances](/tests/object_testbed.ipynb)
- [For inspecting data output from pezEval and baselineEval](/tests/scrape_data.ipynb)
- [Sample trained model and output](/tests/models/voltage_c2/)

### Classes
  - GridLearn 
  - Building
    - Weather
    - HeatPump
    - ElectricHeater
    - EnergyStorage
  - RBC Agent (optional)
  - RL Agent (optional)
![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/images/citylearn_diagram.png)

### Building
The Building class is the main building block of the GridLearn environment.

The DHW and cooling demands of the buildings have been pre-computed and obtained from EnergyPlus. The DHW and cooling supply systems are sized such that the DHW and cooling demands are always satisfied. CityLearn automatically sets constraints to the actions from the controllers to guarantee that the DHW and cooling demands are satisfied, and that the building does not receive from the storage units more energy than it needs. 

The file building_attributes.json contains the attributes of each building, which can be modified. We do not advise to modify the attributes Building -> HeatPump -> nominal_power and Building -> ElectricHeater -> nominal_power from their default value "autosize", as they guarantee that the DHW and cooling demand are always satisfied.

#### Building Parameters
- ```data_path```: path to energy modeling data (string)
- ```climate_zone```: selects a climate file for weather data and climate specific energy modeling output (int)
- ```buildings_states_actions_file```: path to .json file with boolean T/F for each available state and action (string)
- ```hourly_timesteps```: number of timesteps in an hour (e.g. 4 = 15 minute intervals) (int)
- ```uid```: selects a building from the available data in ```buildings_states_actions_file``` and ```data_path``` (string)
- ```weather```: a Weather object
- ```save_memory```: does not log {action selection, state values, rewards} if True (boolean) 

#### Building utility attributes (OpenAI gym compatible attributes and other debugging attributes)
- ```action_space```: a normalized ((-1,1)) Spaces.Box object for all actions available to the controlling agent
- ```observation_space```: a normalized ((-1,1)) Spaces.Box object for all observations available to the controlling agent
- ```normalization_mid```, ```normalization_range```: components for mapping observations to human readable values
- ```rbc```: indicates whether the building is RL or RBC controlled (boolean)
#### Building utility methods  
- ```step()```: uses the ```get_obs()``` and ```get_reward()``` methods to return the (next state, reward, observation, info) tuple with input = action
- ```normalize()```: uses the reward values observed thus far to normalize future reward values
- ```close()```: writes log values to csv -- will be depricated in favor of ```terminate()``` in the future
#### Building energy attributes (Metrics in kWh)
- ```cooling_demand_building```: demand for cooling energy to cool down and dehumidify the building
- ```dhw_demand_building```: demand for heat to supply the building with domestic hot water (DHW)
- ```electric_consumption_appliances```: non-shiftable electricity consumed by appliances
- ```electric_generation```: electricity generated by the solar panels
- ```electric_consumption_cooling```: electricity consumed to satisfy the cooling demand of the building
- ```electric_consumption_cooling_storage```: if > 0, electricity consumed by the building's cooling device (i.e. heat pump) to increase cooling energy stored; if < 0, electricity saved from being consumed by the building's cooling device (through decreasing the cooling energy stored and releasing it into the building's cooling system).
- ```electric_consumption_dhw```: electricity consumed to satisfy the DHW demand of the building
- ```electric_consumption_dhw_storage```: if > 0, electricity consumed by the building's heating device (i.e. DHW) to increase DHW energy storage; if < 0, electricity saved from being consumed by the building's heating device (through decreasing the heating energy stored and releasing it into the building's DHW system).
- ```net_electric_consumption```: building net electricity consumption
- ```net_electric_consumption_no_storage```: building net electricity consumption if there were no cooling and DHW storage
- ```net_electric_consumption_no_pv_no_storage```: building net electricity consumption if there were no cooling, DHW storage and PV generation
- ```cooling_device_to_building```: cooling energy supplied by the cooling device (i.e. heat pump) to the building
- ```cooling_storage_to_building```: cooling energy supplied by the cooling storage device to the building
- ```cooling_device_to_storage```: cooling energy supplied by the cooling device to the cooling storage device
- ```cooling_storage_soc```: state of charge of the cooling storage device
- ```dhw_heating_device_to_building```: DHW heating energy supplied by the heating device to the building
- ```dhw_storage_to_building```: DHW heating energy supplied by the DHW storage device to the building
- ```dhw_heating_device_to_storage```: DHW heating energy supplied by the heating device to the DHW storage device
- ```dhw_storage_soc```: state of charge of the DHW storage device

#### Building energy methods (Controls for energy subsystems)
- ```set_state_space()``` and ```set_action_space()``` set the state-action space of each building
- ```set_storage_heating()``` and ```set_storage_cooling()``` set the state of charge of the ```EnergyStorage``` device to the specified value and within the physical constraints of the system. Returns the total electricity consumption of the building for heating and cooling respectively at that time-step.
- ```get_non_shiftable_load()```, ```get_solar_power()```, ```get_dhw_electric_demand()``` and ```get_cooling_electric_demand()``` get the different types of electricity demand and generation.
- ```set_phase_lag()```: sets the angle between voltage and current
- ```get_solar_power()```: gets the amount of solar generation in kVA
- ```auto_size()``` automatically sizes the heat pumps and the storage devices. It assumes fixed target temperatures of the heat pump for heating and cooling, which combines with weather data to estimate their hourly COP for the simulated period. The ```HeatPump``` is sized such that it will always be able to fully satisfy the heating and cooling demands of the building. This function also sizes the ```EnergyStorage``` devices, setting their capacity as 3 times the maximum hourly cooling demand in the simulated period.

### GridLearn
This class of type OpenAI Gym Environment contains all the buildings and their subclasses.
#### Parameters
- ```model_name```: name for the trained model and output file directory (in [models/](/tests/models)) (string)
- ```data_path```: path to energy modeling data (string)
- ```climate_zone```: selects a climate file for weather data and climate specific energy modeling output (int)
- ```buildings_states_actions_file```: path to .json file with boolean T/F for each available state and action (string)
- ```hourly_timesteps```: number of timesteps in an hour (e.g. 4 = 15 minute intervals) (int)
- ```save_memory = True```: does not log {action selection, state values, rewards} if True (boolean) 
- ```building_ids=None```: a set of values of buildings available in the grid simulation (list)
- ```nclusters=1```: number of clusters that alternate in taking turns (int)
- ```randomseed=2```: random seed for building ordering, random naming, etc (int)
- ```max_num_houses=None```: truncates the number of buildings in the simulation, if none then nhouses is unlimited and is equal to number of buses times the number of buildings per bus (int)
- ```percent_rl=1```: penetration rate of RL agents (1-n) are the number of RBC agents (float)
#### Internal attributes (Power Flow)
- ```net```: a pandapower network 
- ```normalize_reward```: normalizes the reward of all the buildings in the grid
- ```update_grid```: updates the grid based on each building after they implement the building level ```step()``` method
#### Internal attributes (Energy Modeling, all in kWh)
- ```net_electric_consumption```: district net electricity consumption
- ```net_electric_consumption_no_storage```: district net electricity consumption if there were no cooling storage and DHW storage
- ```net_electric_consumption_no_pv_no_storage```: district net electricity consumption if there were no cooling storage, DHW storage and PV generation
- ```electric_consumption_dhw_storage```: electricity consumed in the district to increase DHW energy stored (when > 0) and electricity that the decrease in DHW energy stored saves from consuming in the district (when < 0).
- ```electric_consumption_cooling_storage```: electricity consumed in the district to increase cooling energy stored (when > 0) and electricity that the decrease in cooling energy stored saves from consuming in the district (when < 0).
- ```electric_consumption_dhw```: electricity consumed to satisfy the DHW demand of the district
- ```electric_consumption_cooling```: electricity consumed to satisfy the cooling demand of the district
- ```electric_consumption_appliances```: non-shiftable electricity consumed by appliances
- ```electric_generation```: electricity generated in the district 
#### GridLearn specific methods
- ```get_state_action_spaces()```: returns state-action spaces for all the buildings
- ```get_building_information()```: returns attributes of the buildings that can be used by the RL agents (i.e. to implement building-specific RL agents based on their attributes, or control buildings with correlated demand profiles by the same agent)
- ```get_baseline_cost()```: returns the costs of a Rule-based controller (RBC), which is used to divide the final cost by it.
- ```cost()```: returns the normlized cost of the enviornment after it has been simulated. cost < 1 when the controller's performance is better than the RBC.
#### Methods inherited from OpenAI Gym
- ```step()```: advances simulation to the next time-step and takes an action based on the current state
- ```_get_ob()```: returns all the states
- ```_terminal()```: returns True if the simulation has ended
- ```seed()```: specifies a random seed

### Heat pump
Its efficiency is given by the coefficient of performance (COP), which is calculated as a function of the outdoor air temperature and of the following parameters:

-```eta_tech```: technical efficiency of the heat pump

-```T_target```: target temperature. Conceptually, it is  equal to the logarithmic mean of the temperature of the supply water of the storage device and the temperature of the water returning from the building. Here it is assumed to be constant and defined by the user in the [building_attributes.json](/data/building_attributes.json) file.  For cooling, values between 7C and 10C are reasonable.
Any amount of cooling demand of the building that isn't satisfied by the ```EnergyStorage``` device is automatically supplied by the ```HeatPump``` directly to the ```Building```, guaranteeing that the cooling demand is always satisfied. The ```HeatPump``` is more efficient (has a higher COP) if the outdoor air temperature is lower, and less efficient (lower COP) when the outdoor temperature is higher (typically during the day time). On the other hand, the electricity demand is typically higher during the daytime and lower at night. ```cooling_energy_generated = COP*electricity_consumed, COP > 1```
#### Attributes
- ```cop_heating```: coefficient of performance for heating supply
- ```cop_cooling```:  coefficient of performance for cooling supply
- ```electrical_consumption_cooling```: electricity consumed for cooling supply (kWh)
- ```electrical_consumption_heating```: electricity consumed for heating supply (kWh)
- ```heat_supply```: heating supply (kWh)
- ```cooling_supply```: cooling supply (kWh)
#### Methods
- ```get_max_cooling_power()``` and ```get_max_heating_power()``` compute the maximum amount of heating or cooling that the heat pump can provide based on its nominal power of the compressor and its COP. 
- ```get_electric_consumption_cooling()``` and ```get_electric_consumption_heating()``` return the amount of electricity consumed by the heat pump for a given amount of supplied heating or cooling energy.
### Energy storage
Storage devices allow heat pumps to store energy that can be later released into the building. Typically every building will have its own storage device, but CityLearn also allows defining a single instance of the ```EnergyStorage``` for multiple instances of the class ```Building```, therefore having a group of buildings sharing a same energy storage device.

#### Attributes
- ```soc```: state of charge (kWh)
- ```energy_balance```: energy coming in (if positive) or out (if negative) of the energy storage device (kWh)

#### Methods
- ```charge()``` increases (+) or decreases (-) of the amount of energy stored. The input is the amount of energy as a ratio of the total capacity of the storage device (can vary from -1 to 1). Outputs the energy balance of the storage device.
## Environment variables
The file [buildings_state_action_space.json](/buildings_state_action_space.json) contains all the states and action variables that the buildings can possibly return:
### Possible states
- ```month```: 1 (January) through 12 (December)
- ```day```: type of day as provided by EnergyPlus (from 1 to 8). 1 (Sunday), 2 (Monday), ..., 7 (Saturday), 8 (Holiday)
- ```hour```: hour of day (from 1 to 24).
- ```daylight_savings_status```: indicates if the building is under daylight savings period (0 to 1). 0 indicates that the building has not changed its electricity consumption profiles due to daylight savings, while 1 indicates the period in which the building may have been affected.
- ```t_out```: outdoor temperature in Celcius degrees.
- ```t_out_pred_6h```: outdoor temperature predicted 6h ahead (accuracy: +-0.3C)
- ```t_out_pred_12h```: outdoor temperature predicted 12h ahead (accuracy: +-0.65C)
- ```t_out_pred_24h```: outdoor temperature predicted 24h ahead (accuracy: +-1.35C)
- ```rh_out```: outdoor relative humidity in %.
- ```rh_out_pred_6h```: outdoor relative humidity predicted 6h ahead (accuracy: +-2.5%)
- ```rh_out_pred_12h```: outdoor relative humidity predicted 12h ahead (accuracy: +-5%)
- ```rh_out_pred_24h```: outdoor relative humidity predicted 24h ahead (accuracy: +-10%)
- ```diffuse_solar_rad```: diffuse solar radiation in W/m^2.
- ```diffuse_solar_rad_pred_6h```: diffuse solar radiation predicted 6h ahead (accuracy: +-2.5%)
- ```diffuse_solar_rad_pred_12h```: diffuse solar radiation predicted 12h ahead (accuracy: +-5%)
- ```diffuse_solar_rad_pred_24h```: diffuse solar radiation predicted 24h ahead (accuracy: +-10%)
- ```direct_solar_rad```: direct solar radiation in W/m^2.
- ```direct_solar_rad_pred_6h```: direct solar radiation predicted 6h ahead (accuracy: +-2.5%)
- ```direct_solar_rad_pred_12h```: direct solar radiation predicted 12h ahead (accuracy: +-5%)
- ```direct_solar_rad_pred_24h```: direct solar radiation predicted 24h ahead (accuracy: +-10%)
- ```t_in```: indoor temperature in Celcius degrees.
- ```avg_unmet_setpoint```: average difference between the indoor temperatures and the cooling temperature setpoints in the different zones of the building in Celcius degrees. sum((t_in - t_setpoint).clip(min=0) * zone_volumes)/total_volume
- ```rh_in```: indoor relative humidity in %.
- ```non_shiftable_load```: electricity currently consumed by electrical appliances in kWh.
- ```solar_gen```: electricity currently being generated by photovoltaic panels in kWh.
- ```cooling_storage_soc```: state of the charge (SOC) of the cooling storage device. From 0 (no energy stored) to 1 (at full capacity).
- ```dhw_storage_soc```: state of the charge (SOC) of the domestic hot water (DHW) storage device. From 0 (no energy stored) to 1 (at full capacity).
- ```net_electricity_consumption```: net electricity consumption of the building (including all energy systems) in the current time step.
- ```absolute_voltage```: the per unit voltage at the nearest distribution bus
- ```total_voltage_spread```: the difference between the maximum and minimum voltage in the network
- ```relative_voltage```: the percentile ranking of the nearest bus's voltage

### Possible actions
C determines the capacity of the storage device and is defined as a multiple of the maximum thermal energy consumption by the building.
- ```cooling_storage```: increase (action > 0) or decrease (action < 0) of the amount of cooling energy stored in the cooling storage device. -1/C <= action <= 1/C (attempts to decrease or increase the cooling energy stored in the storage device by an amount equal to the action times the storage device's maximum capacity). In order to decrease the energy stored in the device (action < 0), the energy must be released into the building's cooling system. Therefore, the state of charge will not decrease proportionally to the action taken if the demand for cooling of the building is lower than the action times the maximum capacity of the cooling storage device.
- ```dhw_storage```: increase (action > 0) or decrease (action < 0) of the amount of DHW stored in the DHW storage device. -1/C <= action <= 1/C (attempts to decrease or increase the DHW stored in the storage device by an amount equivalent to action times its maximum capacity). In order to decrease the energy stored in the device, the energy must be released into the building. Therefore, the state of charge will not decrease proportionally to the action taken if the demand for DHW of the building is lower than the action times the maximum capacity of the DHW storage device.
- ```energy_storage```: increase (action > 0) or decrease (action < 0) of the amount of energy stored in a battery. Similarly the action is constrained by the available energy stored in the device and the remaining storage
- ```pv_curtail```: curtailment of solar power production.
- ```pv_phi```: change in phase lag of the inverter. Phi between 0 (completely skewed towards real power) and 90 degrees (completely skewed towards reactive power)

Note for storage actions that the action of the user-implemented controller can be bounded between -1/C and 1/C because the capacity of the storage unit, C, is defined as a multiple of the maximum thermal energy consumption by the building. For instance, if C_cooling = 3 and the peak cooling energy consumption of the building during the simulation is 20 kWh, then the storage unit will have a total capacity of 60 kWh.

The mathematical formulation of the effects of the actions can be found in the methods ```set_storage_heating(action)``` and ```set_storage_cooling(action)``` of the class Building in the file [energy_models.py](/energy_models.py).
### Reward function
The reward function must be defined by the user by changing the ```get_reward()``` method in the file [/citylearn/energy_models.py](/citylearn/energy_models.py?plain=1#L237).

By modifying these functions the user changes the reward that the CityLearn environment returns every time the method .step(a) is called.

### Performance metrics
These components have been depricated in favor of the ```scrape_data.ipynb``` file which can compare across any previously run data file.

#### With special thanks to original CityLearn authors for their support in this project.
[Vázquez-Canteli, J.R., Kämpf, J., Henze, G., and Nagy, Z., "CityLearn v1.0: An OpenAI Gym Environment for Demand Response with Deep Reinforcement Learning", Proceedings of the 6th ACM International Conference, ACM New York p. 356-357, New York, 2019](https://dl.acm.org/citation.cfm?id=3360998)

## License
The MIT License (MIT) Copyright (c) 2021, University of Colorado Boulder

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

