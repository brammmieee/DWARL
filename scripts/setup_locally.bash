#!/bin/bash

# Check if the current system is Ubuntu 22.04 Desktop
if [[ $(lsb_release -rs) == "22.04" && $(lsb_release -is) == "Ubuntu" && $(uname -m) == "x86_64" ]]; then
    echo "Detected Ubuntu 22.04 Desktop. Continuing..."
else
    echo "Python venv install of package is only officially supported on Ubuntu 22.04. Consider using the dockerized version."
    read -p "Do you want to continue anyway? (y/n): " choice
    case "$choice" in 
      y|Y ) echo "Continuing despite the recommendation...";;
      n|N ) echo "Exiting..."; exit 1;;
      * ) echo "Invalid choice. Exiting..."; exit 1;;
    esac
fi

# Perform system update and upgrade
sudo apt update -y
sudo apt upgrade -y

# Installing Webots 2023b
echo "Installing Webots 2023b..."
wget -qO- https://cyberbotics.com/Cyberbotics.asc | sudo apt-key add -
sudo add-apt-repository 'deb https://cyberbotics.com/debian/ binary-amd64/'
sudo apt install -y webots

# Istall Cuda toolkit
sudo apt install nvidia-cuda-toolkit

# Install Python 3 pip and venv
sudo apt install python3-venv
sudo apt install -y python3-pip

# Get the parent directory path
parent_dir=$(dirname "$(pwd)")

# Create Python virtual environment in parent directory
echo "Creating Python virtual environment in $parent_dir..."
python3 -m venv "$parent_dir/venv"

# Environment variable setup to be appended in the virtual environment activation script
echo "
# Custom environment variables for Webots and other settings
export WEBOTS_HOME='/usr/local/webots'
export PYTHONPATH='/usr/local/webots/lib/controller/python:$parent_dir:$parent_dir/venv/lib/python3.10/site-packages'
export PYTHONIOENCODING='UTF-8'
export TF_ENABLE_ONEDNN_OPTS=0
" >> "$parent_dir/venv/bin/activate"

# Activate virtual environment
source "$parent_dir/venv/bin/activate"

# Install PyTorch specifically for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other Python packages from requirements.txt
echo "Installing Python packages..."
pip install -r "$parent_dir/requirements.txt"

# Deactivate virtual environment
deactivate

echo "Python packages installed successfully."
echo "Webots 2023b and environment setup completed."
