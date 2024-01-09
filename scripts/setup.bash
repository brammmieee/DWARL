# Adding package details for running GPU enabled docker container
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo apt update;

# Installing apt packages
sudo apt install -y docker
sudo apt-get install -y nvidia-container-toolkit # enable GPU container

# Build docker image (check if build exists with `docker images`)
docker build -t dwarl_container .. --progress plain