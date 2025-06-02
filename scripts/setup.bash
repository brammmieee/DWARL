# Adding package details for running GPU enabled docker container
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update -y

# # Installing apt packages
# sudo apt install -y docker

# # Fix for non-responsive GPU 
# sudo apt remove nvidia-container-toolkit* -y # Remove other versions
# sudo apt install -y nvidia-container-toolkit-base=1.14.6-1
# sudo apt install -y nvidia-container-toolkit=1.14.6-1  

sudo apt install -y nvidia-container-toolkit

# Configure nvidia docker runtime environment
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
