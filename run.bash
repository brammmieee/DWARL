# Enable connection to Xserver
xhost +local:root > /dev/null 2>&1
# (If you need to disable connections to the X server, you can do it with the following command: xhost -local:root > /dev/null 2>&1.)
docker run --gpus=all -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw dwarl_container