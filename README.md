# DWARL Package

DWARL is a comprehensive package designed for [describe the purpose of the package here]. It provides a robust and flexible framework for [describe what the package does].

## Installation and Setup

The package can either be installed locally with python packages in a venv or containerized. When training on a remote desktop it's recommended to use the local install since the containerized installation struggles with x11 forwarding.

### Local Install

To install the package locally, follow these steps:

1. Run the `setup_locally.py` script from the [scripts folder](./scripts/). This script checks system requirements, installs Webots2023b locally using apt, creates a python venv in the main package folder, and installs the python requirements.
2. To activate the python venv, run `source venv/bin/activate`. 

### Docker install

To install the containerized version of the package, follow these steps:

1. Run `setup.bash` from the [scripts folder](./scripts/). This script installs the required Nvidia container toolkit locally on your system to enable GPU and subsequently builds the docker container as defined in the [docker file](./Dockerfile).
2. To create a docker image from the container build and to subsequently start the container, run the `start.bash` script from the [scripts folder](./scripts/). The docker run command ensures that X11 forwarding is enabled, mounts the host network, enables GPU utilization, enables super user access, and mounts the package folder inside the docker container for ease of development and version control.
3. The docker image can be stopped and subsequently removed with the `stop.bash` script. This also resets the xhost setting to the last permissive standard.

## First Time Usage
### Training

### Testing


## Environment and Environment Wrappers
### Base Environment
### Wrappers


## Utility
### Scripts
attach
...
### Base tooling
### Adminstrative tooling
### Map to Proto Converter
### Wrapper Specific Tooling





