# adding package details for running GPU enabled docker container
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo apt update;

# installing apt packages
sudo apt install -y docker
sudo apt-get install -y nvidia-container-toolkit # enable GPU container

# build image
docker build -t my_custom_webots .