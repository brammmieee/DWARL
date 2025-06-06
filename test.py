# Prevent TensorFlow from spamming messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from environments.base_env import BaseEnv
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from stable_baselines3.ppo import PPO
from utils.data_loader import InfiniteDataLoader
from utils.webots_resource_generator import WebotsResourceGenerator
from utils.wrapper_tools import wrap_env
import hydra
import utils.data_generator as dg
import utils.data_set as ds
import utils.test_tools as tt
from utils.train_tools import validate_config

@hydra.main(config_path='config', config_name='test', version_base='1.1')
def main(cfg: DictConfig):
    # Validate the configuration
    validate_config(cfg)
    
    # Print the output directory if saving is on:
    if cfg.plotter.save:
        print(f"Results will be saved to: {cfg.paths.test_results.results}")

    # Load the training config
    path_to_training_run_output = Path(cfg.paths.outputs.training) / str(cfg.setup.model.date) / str(cfg.setup.model.time)
    path_to_train_cfg = path_to_training_run_output / '.hydra/config.yaml'
    train_cfg = OmegaConf.load(path_to_train_cfg)
    
    # Generate data
    if cfg.generate_data:
        print('Removing old data and generating new...')
        # Generate map, path and data points in our axis convention
        data_generator = dg.DataGenerator(cfg.data_generator, cfg.paths, cfg.seed)
        data_generator.erase_old_data()
        data_generator.generate_data()

        # Generate the proto and world files for the Webots environment from the generated data
        webots_resource_generator = WebotsResourceGenerator(train_cfg.setup.simulation, cfg.paths)
        webots_resource_generator.erase_old_data()
        webots_resource_generator.generate_resources()
                
    # Initialize the dataset and the data loader
    data_set = ds.Dataset(cfg.paths)
    data_loader = InfiniteDataLoader(data_set, num_envs=1, seed=cfg.seed)
    
    # Creating wrapped environment
    env = wrap_env(
        BaseEnv(
            cfg=train_cfg.setup.environment,
            paths=cfg.paths,
            sim_cfg=train_cfg.setup.simulation,
            data_loader=data_loader,
            env_idx=0,
            render=cfg.render,
        ), 
        train_cfg.setup.wrappers
    )
        
    # Load model
    path_to_models = path_to_training_run_output / 'models'
    if int(cfg.setup.model.steps) > 0:
        model = PPO.load(path_to_models / f'rl_model_{cfg.setup.model.steps}_steps.zip', env=env)
    else: # steps < 0 means load the best model
        model = PPO.load(path_to_models / 'best_model.zip', env=env)

    # Evaluate model and plot results
    results = tt.evaluate_model(
        nr_episodes=cfg.nr_episodes, 
        env=env, 
        model=model, 
        max_nr_steps=cfg.max_episode_steps, 
        deterministic=cfg.setup.model.deterministic, 
        seed=cfg.seed
    )
    plotter = tt.ResultPlotter(cfg.plotter)
    plotter.plot_paths(results)
    if cfg.plotter.save:
        plotter.save_plots(cfg.paths.test_results.results, prefix='paths')
    plotter.plot_velocities(results)
    if cfg.plotter.save:
        plotter.save_plots(cfg.paths.test_results.results, prefix='velocities')

    # Save evaluation results
    if cfg.json_results.save:
        tt.save_results(results, cfg.json_results.json_file_path)

    # Prevent figure from closing when script is done running
    if cfg.plotter.show:
        input("Press Enter to exit and close the plots...")

if __name__ == "__main__":
    main()