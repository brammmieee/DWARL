FROM cyberbotics/webots:R2023b-ubuntu22.04

WORKDIR /DWARL
COPY . /DWARL

# Webots - Setting environment variables
ENV PYTHONPATH ${WEBOTS_HOME}/lib/controller/python:/DWARL
ENV PYTHONIOENCODING $PYTHONIOENCODING:UTF-8

# Install pip and required python packages
RUN apt update
RUN apt upgrade -y
RUN apt install -y python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r /DWARL/requirements.txt

# Install OpenCV headless (fix for dependency between matplotlib and gymnasium)
RUN pip uninstall opencv-python
RUN pip install opencv-python-headless