if __name__=='__main__':
    import multiprocessing
    import sys
    from pettingzoo.test import parallel_api_test

    import multiprocessing
    import sys
    from pettingzoo.test import parallel_api_test
    from citylearn import GridLearn
    from citylearn import MyEnv
    from pathlib import Path
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3 import PPO
    import gym

    import multiprocessing
    import sys
    import supersuit as ss

    import time

    # multiprocessing.set_start_method("fork")

    climate_zone = 1
    data_path = Path("../citylearn/data/Climate_Zone_"+str(climate_zone))
    buildings_states_actions = '../citylearn/buildings_state_action_space.json'

    config = {
        "data_path":data_path,
        "climate_zone":climate_zone,
        "buildings_states_actions_file":buildings_states_actions,
        "hourly_timesteps":4
    }

    grid = GridLearn(**config)

    envs = [MyEnv(grid), MyEnv(grid)]

    print('padding action/observation spaces...')
    envs = [ss.pad_action_space_v0(env) for env in envs]
    envs = [ss.pad_observations_v0(env) for env in envs]

    print('creating pettingzoo env...')
    envs = [ss.pettingzoo_env_to_vec_env_v0(env) for env in envs]

    print('stacking vec env...')
    nenvs = 2
    envs = [ss.concat_vec_envs_v0(env, nenvs, num_cpus=1, base_class='stable_baselines3') for env in envs]

    print('setting the grid...')
    for env in envs[1:]:
        for n in range(nenvs):
            env.venv.vec_envs[n].par_env.aec_env.env.env.env.env.set_grid(envs[0].venv.vec_envs[n].par_env.aec_env.env.env.env.env.grid)

    models = [PPO(MlpPolicy, env, verbose=2, gamma=0.999, batch_size=2, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1) for env in envs]

    for _ in range(5):
        for model in models:
            model.learn(1, reset_num_timesteps=False)

    obss = [env.reset() for env in envs]
    for _ in range(5):
        for m in range(len(models)):
            obss[m] = [np.concatenate(list(envs[m].venv.vec_envs[i].par_env.state(i).values())) for i in range(len(envs[m].venv.vec_envs))]
            obss[m] = np.concatenate(obss[m])
            obss[m] = obss[m].reshape(-1,1)

            action = models[m].predict(obss[m])[0]
            obss[m], reward, done, info = envs[m].step(action)
