# DWARL Package

DWARL is a comprehensive package designed for creating controller policies through simulation-based learning.

## Installation and Setup

The package can be installed locally with Python packages in a virtual environment or containerized (RECOMMENDED).

### Docker Installation (Recommended for Local Use)

To install the containerized version of the package, make sure you have a priviliged docker install before following the following steps:

1. Run `setup.bash` **from the [scripts folder](./scripts/)**. This script installs the required Nvidia container toolkit locally on your system to enable GPU usage.
2. Run the `run.bash` script **from the [scripts folder](./scripts/)**. This script will pull the docker image from docker-hub and ensures that X11 forwarding is enabled, mounts the host network, enables GPU utilization, grants superuser access, and mounts the package folder inside the Docker container for ease of development and version control.
3. To stop and remove the Docker image, run the `stop.bash` script. This also resets the `xhost` setting to the last permissive standard.

To attach to the Docker container, run the `attach.bash` script from the [scripts folder](./scripts/).

### Local Installation (Recommended for Remote Systems)

To install the package locally, follow these steps:

1. Run the `setup_locally.py` script **from the [scripts folder](./scripts/)**. This script checks system requirements, installs Webots2023b locally using `apt`, creates a Python venv in the main package folder, and installs the Python requirements.

3. To activate the Python venv, run `source venv/bin/activate`.

### Known Issues
When installing the package locally according to the [section](#local-installation-recommended) above, the issue might occur that upon creating the virtual environment, no activate script is created within the 'bin' folder of your newly created venv. To fix this issue, remove the 'venv' folder completely, reinstall python3-venv by running `sudo apt purge python3-venv` and `sudo apt install python3-venv`, after which you can try running the local install script again.

## Training & Testing

In this section the training and testing scripts are explained. For configuring a training or testing run, [Hydra](https://hydra.cc/) is used. More on configuration in the [configuration](#configurations) section below.

For a graphical overview of the testing and training pipeline, please have a look at [implementation slides](./docs/DWARL%20Implementation%20Slides.pdf). To get an idea of what the capabilities are of the framework, have a look at the commands and their explanation in the [demo commands file](./demo_commands.bash).

### Training

To start training, run `train.py`. To view the available arguments, run `train.py -h`. The script will automatically display missing arguments.

The training script will create an output folder in the [main directory](./outputs) with a sub-directory called "training", where it will generate subfolders for logs, models, configs and configuration overwrites within a date-time based folder structure.
- **logs**: Contains TensorBoard training logs.
- **models**: Stores the trained models saved according to the callback parameters passed to the training script.
- **.hydra**: Houses the configurations used to train the network, the overwrites from the command line and the configuration used for the Hydra configuration engine.

#### Plotting Training Results

To plot the training results, run `tensorboard --logdir "log directory"`, where "log directory" is the relative path to the log directories created during the training run. The output of the command will provide a web interface URL to view the results on TensorBoard.

### Testing

To start testing, run `test.py`. To view the available arguments, run `test.py -h`. The script will automatically display missing arguments.

This script lets you import a model based on the date-time, and number of steps (or -1 for the model with the most amount of steps). It then evaluates the network by running through the environment and saving results. The results are saved within the "testing" sub-directory of the outputs folder where, like with train.py, a date-time based folder structure is created. 

Like with training, the script saves the configuration used for testing. The results are stored in the form of a json file and an optional image file with the results visualized in terms of position and velocity. Here you can see an example of the [position output](./docs/Example%20output%20on%20path%20from%20test%20script.png) and [velocity output](./docs/Example%20output%20on%20velocity%20from%20test%20script.png).

## Data generation

Future work!

## Environment and Environment Wrappers

### Base Environment

`BaseEnv` is a [custom Gym environment](https://gymnasium.farama.org/) designed for robotic simulations using Webots and Gymnasium. It serves as a base class for creating and managing complex robotic simulations, providing essential methods for initialization, resetting, stepping through actions, and rendering.

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

The standard configuration can be configured through the yaml files that are parsed from the [config](./config) folder. Parsing is performed by the Hydra configuration engine. 

More detailed information on the existing parameters is left for future work.

### Proto Configs

Proto configs allow you to change the configuration of the proto files used by Webots to create the simulation.

- **Proto Config**: The base environment loads the Webots world file, which imports the robot proto and map protos. The robot proto includes the LIDAR sensor proto, which can be custom-configured.
    - Note: When adding a custom config that changes any proto properties, save the default setup in the `default_proto_config` to revert to the base configuration.

Add template files with `{{key}}` items defined by the keys in the substitutions dictionary under the custom config. See the proto_configs folder for examples. Add non-template files to the `.gitignore`.

## Notes on Notation and Conventions

In this section, we provide documentation for the notation and conventions used in the codebase. This documentation ensures consistency and enhances code comprehension and extensibility. We adhere to SI units throughout the entire package, utilizing meters (m), meters per second (m/s), radians (rad), radians per second (rad/s), and other SI units.

### Axis Systems

In this system, the x-axis points to the right of the robot, the y-axis points forward, and positive rotation around the z-axis is clockwise. This choice is made to facilitate reasoning about similarities between Euclidean space and velocity space.

### Vector Entries

The vector entries follow the following order. Note that the linear velocity v (velocity in the robot's positive y direction) is the second entry in the velocity vector to simplify plotting.

- **XY Space**: `[x, y]`
- **Velocities**: `[omega, v]`
- **Polar Coordinates**: `[angle, dist]`

## Utilities

Check out the utilities [here](./utils/):

- **Admin Tools**: General tooling module, not necessarily specific to this package (e.g., logging setup, configuration parsing).
- **Base Tools**: Tooling module for setting up and using the base environment (e.g., environment initialization, data processing).
- **Wrapper Tools**: Modules specific to wrapper functionality.

For further details and examples, refer to the respective folders and documentation links provided.

## Tab Completion
Run the following command in the package folder to enable tab completion:s
```bash
eval "$(python train.py -sc install=bash)"; eval "$(python3 train.py -sc install=bash)"; eval "$(python test.py -sc install=bash)"; eval "$(python3 test.py -sc install=bash)"
```

## Continue Training Example
```bash
python3 train.py quit_sim=true generate_data=true steps=50000 model=load_model model.date=2024-11-13 model.time=16-37-08 model.steps=10000
```