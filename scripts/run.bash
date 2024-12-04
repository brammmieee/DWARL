# Enable connection to Xserver
xhost +local:root

# Fix for X11 connection refused
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH

# Run docker container
docker run --name my_dwarl_container -p 8888:8888 --gpus all --privileged -it --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $XAUTH:$XAUTH \
  -e XAUTHORITY=$XAUTH \
  -v "$(pwd)/..:/DWARL" \
  -d dwarl_container