# Disable connections to the X server
xhost -local:root > /dev/null 2>&1

# Stop the container (check if container persists with `docker ps`)
docker stop my_dwarl_container