### BUILD TIME ###
FROM cyberbotics/webots:R2023b-ubuntu22.04

# Set the working directory to /app (you can change this to your preference)
WORKDIR /DWARL

# Copy your application code into the container (if applicable)
COPY . /DWARL

# # Install additional Python packages
# RUN apt-get update && \
#     apt-get install -y python3-pip && \
#     pip3 install -r requirements.txt

# # Set any environment variables as needed
# ENV WEBOTS_HOME=/usr/local/webots
# ENV PYTHONPATH=$PYTHONPATH:/usr/local/webots/lib/controller/python
# ENV PYTHONIOENCODING=$PYTHONIOENCODING:UTF-8

### RUNTIME ###
CMD ["webots"]