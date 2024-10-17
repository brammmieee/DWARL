#!/usr/bin/python3

from environments.base_env import BaseEnv
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
import hydra
import torch as th
import utils.wrapper_tools as wt

@hydra.main(config_path='../config', config_name='train')
def main(cfg : DictConfig):
    # Environment setup
    base_env = BaseEnv(cfg.environment, cfg.paths)
    wrapped_env = wt.wrap_env(cfg.wrappers, base_env)
    vec_env=make_vec_env( #NOTE: Wraps the environment in a Monitor wrapper to have additional training information
        env_id=wrapped_env,
        wrapper_class=lambda env: bt.chain_wrappers(env, wrapper_classes),
        n_envs=cfg.envs, 
        vec_env_cls=SubprocVecEnv, 
    )

    # Model setup
    model=PPO(
        env=vec_env,
        policy=cfg.model.policy_type,
        tensorboard_log=cfg.paths.outputs.logs,
        policy_kwargs={
            'net_arch': cfg.model.net_arch,
            'activation_fn': getattr(th.nn, cfg.model.activation_fn)
        }
    )

    # Training
    model.learn(
        total_timesteps=cfg.steps,
        callback=[
            CheckpointCallback(
                save_freq=cfg.callbacks.model_eval_freq,
                save_path=cfg.outputs.models,
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=0,
            ),
            EvalCallback(
                eval_env=vec_env,
                callback_on_new_best=None,
                callback_after_eval=None,
                n_eval_episodes=cfg.callbacks.model_n_eval_episodes,
                eval_freq=cfg.callbacks.model_eval_freq,
                log_path=None,
                best_model_save_path=cfg.ouputs.models,
                deterministic=False,
                render=False,
                verbose=0,
                warn=True,
            )
        ],
        log_interval=cfg.callbacks.log_interval,
        reset_num_timesteps=False,
        progress_bar=True
    )

if __name__ == '__main__':
    main()

#### Continue Training Stuff #### TODO
# def load_init_model(args):
#     if args.init_model_datetime is None or args.init_model_steps is None:
#         raise ValueError("When using an initial model, both the datetime and steps must be provided.")
#     package_dir = os.path.abspath(os.pardir)
#     model_dir=os.path.join(package_dir,'training', 'archive', 'models', args.init_model_datetime)
#     model_name=f'rl_model_{args.init_model_steps}_steps.zip'
#     model_path=os.path.join(model_dir, model_name)
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model directory {model_path} does not exist, continuing training is not possible.")
#     return PPO.load(model_path)

    # # Function to get values when simply wanting to continue training from latest model
    # if cfg.continue_training and (cfg.init_model_datetime is None or cfg.init_model_steps is None):
    #     latest_model_dir=at.get_latest_model_dir()
    #     cfg.init_model_datetime=latest_model_dir.split('/')[-1]
    #     cfg.init_model_steps=at.get_latest_model_steps(latest_model_dir)
    #     # TODO: load other parameters from archive to continue training without setting envs, steps, etc.


    # if not args.use_init_model:
        # model stuff
    # else:
    #     model=load_init_model(args)
    #     model.set_env(vec_env)
    #     model.tensorboard_log=log_dir

### Force quit which is not working anyways ###
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     bt.killall_webots() # TODO use the env.close() method instead
    #     sys.exit(0)