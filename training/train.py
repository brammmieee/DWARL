#!/usr/bin/python3

from environments.custom_env import CustomEnv
import utils.custom_tools as ct

from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

def main():
    # Environment settings
    n_envs = 10
    env_render_mode = None
    env_wb_open = True
    env_wb_mode = 'training'
    
    # Model settings
    test_nr_today = 0
    comment = '01_09_24_test'
    model_name = ct.get_file_name_with_date(test_nr_today, comment)
    
    model_save_freq = 1000 # [steps]
    model_eval_freq = 10000 # [steps]
    model_n_eval_episodes = 25 
    
    # Training settings
    total_training_steps = 1e4
    
    # Creating vectorized environment
    vec_env = make_vec_env(
        env_id=CustomEnv, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            'render_mode': env_render_mode, 
            'wb_open': env_wb_open, 
            'wb_mode': env_wb_mode,
        }
    )
    
    # Creating PPO model with callbacks
    model = PPO(
        policy=MultiInputPolicy,
        env=vec_env,
        tensorboard_log = "./logs/" + model_name,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq = model_save_freq,
        save_path = "./models/" + model_name,
        name_prefix = model_name,
        save_replay_buffer = False,
        save_vecnormalize = False,
        verbose = 0,
    )

    eval_callback = EvalCallback(
        eval_env = vec_env,
        callback_on_new_best = None,
        callback_after_eval = None,
        n_eval_episodes = model_n_eval_episodes,
        eval_freq = model_eval_freq,
        log_path = None,
        best_model_save_path = "./models",
        deterministic = False,
        render = False,
        verbose = 0,
        warn = True,
    )

    # Train model
    model.learn(
        total_timesteps=total_training_steps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        tb_log_name=model_name,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
if __name__=='__main__':
    main()