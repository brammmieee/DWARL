from pathlib import Path
from stable_baselines3.common.monitor import Monitor
import importlib
import os
import wrappers

def load_wrapper_classes(class_names):
    """ Load the wrapper classes from the wrappers module. """
    loaded_classes = []
    wrapper_dir = Path(wrappers.__path__[0])
    for classname in class_names:
        for path_to_wrapper in wrapper_dir.iterdir():
            file_name = path_to_wrapper.name
            
            if file_name.endswith(".py") and file_name != "__init__.py":
                module_name = file_name[:-3]
                module = importlib.import_module(f"wrappers.{module_name}")
                
                if hasattr(module, classname):
                    loaded_classes.append(getattr(module, classname))
                    break
                
    return loaded_classes

def wrap_env(env, cfg):
    """ Wrap the environment with the wrappers specified in the configuration. """    
    class_names = cfg.keys()
    loaded_classes = load_wrapper_classes(class_names)
    
    # Chain the wrappers
    for wrapper_class in loaded_classes:
        wrapper_cfg = cfg.get(wrapper_class.__name__)
        env=wrapper_class(env, wrapper_cfg)
        
    return env

def make_vec_env(
        env_id,
        n_envs=1,
        seed=None,
        start_index=0,
        monitor_dir=None,
        wrapper_class=None,
        env_kwargs=None,
        vec_env_cls=None,
        vec_env_kwargs=None,
        monitor_kwargs=None,
        wrapper_kwargs=None,
):
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int):
        def _init():

            env = env_id(**env_kwargs, env_idx=rank)

            if seed is not None:
                env.action_space.seed(seed + rank)
            
            # Wrap the env in a Monitor wrapper
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    
    return vec_env