# Enable connection to Xserver
xhost +local:root

# Fix for X11 connection refused (https://forums.docker.com/t/x11-over-remote-docker-container-stops-working-after-resetting-vpn/95484)
SOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH

# Run docker container
docker run --name my_dwarl_container -p 8888:8888 --gpus all --privileged -it --network=host  -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$(pwd)/..:/DWARL" -d dwarl_container