from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import VecNormalize
from pathlib import Path
import os
import shutil
import sys

def validate_config(cfg: DictConfig):
    try:
        # This will raise an error if there are any MISSING values
        OmegaConf.to_container(cfg, throw_on_missing=True)
    except Exception as e:
        print("Error: Configuration is incomplete. Please provide values for all required parameters.")
        print(f"Details: {str(e)}")
        # Get the Hydra config to access the output directory
        path_to_output = Path.cwd()
        if path_to_output.exists():
            print(f"Removing output directory: {path_to_output}")
            shutil.rmtree(path_to_output, ignore_errors=True)
        sys.exit(1)
        
def make_vec_env(
        env_class,
        n_envs,
        wrapper_class,
        seed,
        env_kwargs,
        vec_env_cls,
        wrapper_kwargs,
        monitor_dir=None,
        monitor_kwargs=None,
        # normalize_kwargs=None,
):
    monitor_kwargs = monitor_kwargs or {}
    # normalize_kwargs = normalize_kwargs or {}

    def make_env(rank):
        def _init():
            env = env_class(**env_kwargs, env_idx=rank)
            
            # Wrap the env in a Monitor wrapper
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            # env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            
            # Wrap the environment with the provided wrapper and seed the action space (defined in the actionwrapper)
            env = wrapper_class(env, **wrapper_kwargs)
            if seed is not None:
                env.action_space.seed(seed + rank)
            
            return env

        return _init

    vec_env = vec_env_cls([make_env(env_idx) for env_idx in range(n_envs)])
    
    # Normalize
    # vec_env = VecNormalize(vec_env, **normalize_kwargs)

    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    
    return vec_env