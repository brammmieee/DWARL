import hydra
from omegaconf import DictConfig
import sys
from environments.base_env import BaseEnv
from stable_baselines3.ppo import PPO
import utils.test_tools as tt
from utils.data_loader import InfiniteDataLoader
import utils.data_set as ds
from utils.wrapper_tools import wrap_env

@hydra.main(config_path='config', config_name='test', version_base='1.1')
def main(cfg: DictConfig):
    # Choose the netwerk to evaluate
    date_time = '24_09_24__17_24_45'
    training_steps = 92320000  # Set the number of training steps (optional)

    # Evaluation parameters
    deterministic = False  # Set whether to use deterministic policy or not
    nr_maps = 25
    max_nr_steps = 1000  # Evaluation will stop after this number of steps
    
    # Load dataset
    data_set = ds.Dataset(cfg.paths)
    data_loader = InfiniteDataLoader(data_set, cfg.envs)
    # TODO generate data for testing specific with split on map level
    
    # Create environment
    env=BaseEnv(
        cfg=cfg.environment,
        paths=cfg.paths,
        sim_cfg=cfg.simulation,
        data_loader=data_loader,
        render_mode=None,
        evaluate=True
    )
    env = wrap_env(env, cfg.wrappers)
    
    # Load model
    if cfg.model.use_best:
        model = PPO.load(, env=env)
    # Load model based on training steps
    else:
        model = PPO.load(, env=env)

    map_name_list = 
    results = tt.evaluate_model(map_name_list, env, model, cfg.max_nr_steps, cfg.mode.deterministic, cfg.seed)
    plotter = tt.ResultPlotter(cfg.result_plotter)
    plotter.plot_results(results)
    
    # TODO: 
    # - decide on output folder to save plots and or results
    # - decide on how to load the training configs
    


    # Save evaluation results
    output_folder = os.path.join(package_dir, 'DWARL', 'testing', 'results', f'{date_time}')
    os.makedirs(output_folder, exist_ok=True)
    json_file_path = os.path.join(output_folder, f'eval_results_{training_steps}_steps.json')
    save_eval_results(eval_results, json_file_path)

    # Load evaluation results
    date_time = '24_09_24__17_24_45'
    training_steps = 92320000  # Set the number of training steps (optional)
    package_dir = os.path.abspath(os.pardir)
    output_folder = os.path.join(package_dir, 'DWARL', 'testing', 'results', f'{date_time}')
    json_file_path = os.path.join(output_folder, f'eval_results_{training_steps}_steps.json')
    eval_results2 = load_eval_results(json_file_path)
    plot_eval_results(eval_results2)
#

if __name__ == "__main__":
    main()
    
    






#
