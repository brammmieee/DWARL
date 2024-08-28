# DWARL Package

DWARL is a comprehensive package designed for creating controller policies through simulation-based learning.

## Installation and Setup

The package can be installed locally with Python packages in a virtual environment (venv) or containerized. For training on a remote desktop, it is recommended to use the local installation due to issues with X11 forwarding in containerized installations.

### Local Installation (Recommended)

To install the package locally, follow these steps:

1. Run the `setup_locally.py` script from the [scripts folder](./scripts/). This script checks system requirements, installs Webots2023b locally using `apt`, creates a Python venv in the main package folder, and installs the Python requirements.
2. To activate the Python venv, run `source venv/bin/activate`.
3. Restart VSCode and add the proper Python path to the [vscode config json file](./.vscode/settings.json) in order to use the interactive scripts. When running an interactive script vscode might prompt you to install the interactive kernell, please do so.

### Docker Installation

To install the containerized version of the package, follow these steps:

1. Run `setup.bash` from the [scripts folder](./scripts/). This script installs the required Nvidia container toolkit locally on your system to enable GPU usage and builds the Docker container as defined in the [Dockerfile](./Dockerfile).
2. To create a Docker image from the container build and subsequently start the container, run the `start.bash` script from the [scripts folder](./scripts/). This script ensures that X11 forwarding is enabled, mounts the host network, enables GPU utilization, grants superuser access, and mounts the package folder inside the Docker container for ease of development and version control.
3. To stop and remove the Docker image, run the `stop.bash` script. This also resets the `xhost` setting to the last permissive standard.

To attach to the Docker container, run the `attach.bash` script from the [scripts folder](./scripts/).

### Known Issues
When installing the package locally according to the [section](#local-installation-recommended) above, the issue might occur that upon creating the virtual environment, no activate script is created within the 'bin' folder of your newly created venv. To fix this issue, remove the 'venv' folder completely, reinstall python3-venv by running `sudo apt purge python3-venv` and `sudo apt install python3-venv`, after which you can try running the local install script again.

## Training

To start training, run `train.py`. To view the available arguments, run `train.py -h`.

The training script will create an archive folder in the [training folder](./training/), where it will generate subfolders for logs, models, and configs:
- **logs**: Contains TensorBoard training logs.
- **models**: Stores the trained models saved according to the callback parameters passed to the training script.
- **configs**: Houses the configurations used to train the network.

### Plotting Training Results

To plot the training results, run `tensorboard --logdir "log directory"`, where "log directory" is the relative path to the log directory created during the training run. The output of the command will provide a web interface URL to view the results on TensorBoard.

## Testing

The testing folder contains several interactive scripts that allow you to run code cells in VSCode using the IPython interpreter, enabling interactive plotting with Matplotlib and easy debugging.

### `evaluate.py`

This interactive script lets you import a model based on the date, time, and number of steps as shown in the model's file name. It then evaluates the network by running through the environment and saving the obtained rewards. More information on the evaluation helper function can be found [here](https://stable-baselines.readthedocs.io/en/master/common/evaluation.html).

### `check.py`

This interactive script facilitates easy debugging, particularly when implementing a custom wrapper. It also allows checking the wrapped environment through the `check_env` utility provided by Stable-Baselines3.

## Environment and Environment Wrappers

### Base Environment

`BaseEnv` is a custom Gym environment designed for robotic simulations using Webots and Gymnasium. It serves as a base class for creating and managing complex robotic simulations, providing essential methods for initialization, resetting, stepping through actions, and rendering.

**Key Features:**
- **Initialization**: Loads configuration files, sets up directories, and initializes Webots simulation.
- **Reset**: Resets the environment, including the map, robot position, and simulation state.
- **Step**: Executes actions, updates the robot's state, computes rewards, and checks for termination conditions.
- **Rendering**: Supports multiple render modes (full, position, velocity, trajectory) to visualize different aspects of the simulation.

### Wrappers

Wrappers allow you to define custom observation, action, and reward wrappers around the base environment. Examples are available, and more information can be found in the [wrapper documentation](https://gymnasium.farama.org/api/wrappers/).

**Observation Wrappers Must:**
- Set the environment's `observation_space` in the wrapper's `__init__` method.

**Action Wrappers Must:**
- Set the environment's `action_space` in the wrapper's `__init__` method.

**Reward Wrappers Must:**
- Implement a proper reward function (BaseEnv always returns 0).

## Configurations

The base environment can be configured through the [base_parameters.yaml](./parameters/base_parameters.yaml) file. This file includes settings like goal tolerance, robot footprint, and goal position sampling behavior.

### Map Numbers

The [map numbers folder](./parameters/map_nrs/) includes the map numbers used for training and testing. The [scripts folder](./scripts/) contains two additional scripts (`change_map_res.py` and `create_map_nrs.py`) for adjusting the map number JSON files.

The maps are obtained from the [BARN dataset](https://www.cs.utexas.edu/~xiao/BARN/BARN.html), which contains 300 maps ranked by difficulty. These maps are converted by the `convert_map2proto` utility script in the [utils folder](./utils/) to be importable in Webots.

### Proto Configs

Proto configs allow you to change the configuration of the proto files used by Webots to create the simulation.

- **Proto Config**: The base environment loads the Webots world file, which imports the robot proto and map protos. The robot proto includes the LIDAR sensor proto, which can be custom-configured.
    - Note: When adding a custom config that changes any proto properties, save the default setup in the `default_proto_config` to revert to the base configuration.

Add template files with `{{key}}` items defined by the keys in the substitutions dictionary under the custom config. See the proto_configs folder for examples. Add non-template files to the `.gitignore`.

## Notes on Notation and Conventions

In this section, we document the notation and conventions used throughout the codebase. This helps maintain consistency and makes the code easier to understand and extend.

### Vector Entries

- **XY Space**: `[x, y]`
- **Velocities**: `[omega, v]`
- **Polar Coordinates**: `[angle, dist]`

This convention is used to make plotting more straightforward.

## Utilities

Check out the utilities [here](./utils/):

- **Admin Tools**: General tooling module, not necessarily specific to this package (e.g., logging setup, configuration parsing).
- **Base Tools**: Tooling module for setting up and using the base environment (e.g., environment initialization, data processing).
- **Wrapper Tools**: Modules specific to wrapper functionality.

For further details and examples, refer to the respective folders and documentation links provided.
