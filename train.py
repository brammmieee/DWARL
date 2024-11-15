# Prevent TensorFlow from spamming messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environments.base_env import BaseEnv
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from utils.data_loader import InfiniteDataLoader
from utils.webots_resource_generator import WebotsResourceGenerator
from utils.wrapper_tools import wrap_env
import hydra
import subprocess
import torch as th
import utils.data_generator as dg
import utils.data_set as ds
import utils.train_tools as tt

@hydra.main(config_path='config', config_name='train', version_base='1.1')
def main(cfg: DictConfig):
    # Validate the configuration
    tt.validate_config(cfg)
      
    # Kill all the Webots processes that are running
    if cfg.quit_sim:
        print('Freeing up resources by killing all Webots instances...')
        subprocess.run(["bash", str(Path(cfg.paths.scripts.killall_webots))])
        
    # Load the parameters for the environment, simulation and wrappers
    load_model = 'environment' not in cfg.setup.keys()
    if load_model:
        print('Loading model setup...')
        # Load the training config
        path_to_training_run_output = Path(cfg.paths.outputs.training) / str(cfg.setup.model.date) / str(cfg.setup.model.time)
        path_to_train_cfg = path_to_training_run_output / '.hydra/config.yaml'
        loaded_cfg = OmegaConf.load(path_to_train_cfg)
    
        # Merging loaded cfg with the new cfg and saving to the output directory
        cfg = OmegaConf.merge(loaded_cfg, cfg) # later args values overwrite earlier args values
        output_path = Path.cwd() / '.hydra/config.yaml'
        OmegaConf.save(cfg, output_path)
    
    # Generate data
    if cfg.generate_data:
        print('Removing old data and generating new...')
        # Generate map, path and data points in our axis convention
        data_generator = dg.DataGenerator(cfg.data_generator, cfg.paths, cfg.seed)
        data_generator.erase_old_data()
        data_generator.generate_data()
        
        # Generate the proto and world files for the Webots environment from the generated data
        webots_resource_generator = WebotsResourceGenerator(cfg.setup.simulation, cfg.paths)
        webots_resource_generator.erase_old_data()
        webots_resource_generator.generate_resources()

    # Initialize the dataset and the data loader
    data_set = ds.Dataset(cfg.paths)
    data_loader = InfiniteDataLoader(data_set, cfg.setup.model.envs)
    
    # Create the environment
    nr_steps = cfg.steps # Set here so hydra can warn about not being set before creating the environment
    vec_env=tt.make_vec_env( #NOTE: Adds the monitor wrapper which might lead to issues with time limit wrapper! (see __init__ description)
        env_class=BaseEnv,
        n_envs=cfg.setup.model.envs,
        seed=cfg.seed, 
        vec_env_cls=SubprocVecEnv, 
        wrapper_class=wrap_env,
        env_kwargs={
            'cfg': cfg.setup.environment,
            'paths': cfg.paths,
            'sim_cfg': cfg.setup.simulation,
            'data_loader': data_loader,
            'render': False,
        },
        wrapper_kwargs={
            'cfg': cfg.setup.wrappers,
        }
    )
    
    # Initialize the model
    if load_model:
        print('Loading initial model...')
        path_to_models = path_to_training_run_output / 'models'
        model = PPO.load(
            path=path_to_models / f'rl_model_{cfg.setup.model.steps}_steps.zip',
            env=vec_env,
        )
        model.tensorboard_log = cfg.paths.outputs.logs
        model.set_env(vec_env)
    else:
        print('Initializing new model...')
        model = PPO(
            env=vec_env,
            policy=cfg.setup.model.policy_type,
            tensorboard_log=cfg.paths.outputs.logs,
            policy_kwargs={
                'net_arch': OmegaConf.to_container(cfg.setup.model.net_arch),
                'activation_fn': getattr(th.nn, cfg.setup.model.activation_fn)
            }
        )

    # Training
    model.learn(
        total_timesteps=nr_steps,
        callback=[
            CheckpointCallback(
                save_freq=max(cfg.callbacks.model_eval_freq // cfg.setup.model.envs, 1),
                save_path=cfg.paths.outputs.models,
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=0,
            ),
            EvalCallback(
                eval_env=vec_env,
                callback_on_new_best=None,
                callback_after_eval=StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=cfg.callbacks.max_no_improvement_evals,
                    min_evals=cfg.callbacks.min_evals,
                    verbose=1
                ),
                n_eval_episodes=cfg.callbacks.model_n_eval_episodes,
                eval_freq=max(cfg.callbacks.model_eval_freq // cfg.setup.model.envs, 1),
                log_path=None,
                best_model_save_path=cfg.paths.outputs.models,
                deterministic=False,
                render=False,
                verbose=0,
                warn=True,
            ),
        ],
        log_interval=cfg.log_interval,
        reset_num_timesteps=False,
        progress_bar=True
    )

if __name__ == '__main__':
    main()