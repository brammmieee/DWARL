import wrappers
from pathlib import Path
import importlib

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

def wrap_env(env, wrapper_cfg):
    """ Wrap the environment with the wrappers specified in the configuration. """    
    class_names = wrapper_cfg.keys()
    loaded_classes = load_wrapper_classes(class_names)
    
    # Chain the wrappers
    for wrapper_class in loaded_classes:
        cfg = wrapper_cfg.get(wrapper_class.__name__)
        env=wrapper_class(env, cfg)
        
    return env