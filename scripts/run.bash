# Enable connection to Xserver
xhost +local:root > /dev/null 2>&1

# Run docker container
docker run --name my_dwarl_container --gpus=all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$(pwd)/..:/DWARL" -d dwarl_container
