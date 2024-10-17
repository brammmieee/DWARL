

def chain_wrappers(env, wrapper_classes):
    for wrapper_class in wrapper_classes:
        env=wrapper_class(env)
        
    return env

def wrap_env(cfg, base_env):
    """
    Wrap the environment with the wrappers specified in the configuration.
    Args:
        cfg: The wrappers configuration dictionary.
    """
    # Load the wrappers configuration
    class_names = cfg.wrappers.keys()

    # Load the classes from the wrappers module
    loaded_classes = {}
    wrapper_dir = wrappers.__path__[0]
    for filename in os.listdir(wrapper_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            print(f"Loading module: {module_name}")
            module = importlib.import_module(f"environments.wrappers.{module_name}")
            
            # Try to get the classes from the module
            for classname in class_names:
                print(f"Checking for class: {classname}")
                if hasattr(module, classname):
                    loaded_classes[classname] = getattr(module, classname)


    wrapper_params = {}
    for wrapper_class in cfg.env.wrappers:
        if hasattr(globals()[wrapper_class], 'params_file_name'):
            wrapper_file_name = globals()[wrapper_class].params_file_name 
            wrapper_params.update({wrapper_file_name: at.load_parameters([wrapper_file_name])})