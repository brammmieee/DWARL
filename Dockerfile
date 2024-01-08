### BUILD TIME ###
FROM cyberbotics/webots:R2023b-ubuntu22.04

# Set the working directory to /app (you can change this to your preference)
WORKDIR /DWARL

# Copy your application code into the container (if applicable)
COPY . /DWARL

# Install Python package manager packages
RUN apt update
RUN apt upgrade -y
RUN apt install -y python3-pip

# Installing required packages
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r /DWARL/requirements.txt