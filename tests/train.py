from utils import make_envs, make_models, train_models, load_models, eval_models
from pathlib import Path
from citylearn import GridLearn

model_name = 'scaled_cubic'
mode = 'train'

climate_zone = 1
data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
buildings_states_actions = '../citylearn/buildings_state_action_space.json'

config = {
    "model_name":model_name,
    "data_path":data_path,
    "climate_zone":climate_zone,
    "buildings_states_actions_file":buildings_states_actions,
    "hourly_timesteps":4,
    "percent_rl":1,
    "nclusters":2,
    "max_num_houses":4
    # "max_num_houses":4
}

grid = GridLearn(**config)

envs = make_envs(grid, config['nclusters'])
if mode == 'train':
    models = make_models(envs)
    train_models(models, model_name)

models = load_models(envs, model_name)
eval_models(models, envs)
