# Disable connections to the X server
xhost -local:root > /dev/null 2>&1

# Stop and remove`` the container and freeing up memory (check if container persists with `docker ps`)
docker stop my_dwarl_container
# docker rm my_dwarl_container