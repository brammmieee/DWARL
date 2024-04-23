FROM cyberbotics/webots:R2023b-ubuntu22.04

WORKDIR /DWARL
COPY . /DWARL

# Webots - Run the shell script installing linux runtime dependencies
RUN chmod +x /DWARL/scripts/webots_linux_runtime_dependencies.sh
RUN /DWARL/scripts/webots_linux_runtime_dependencies.sh

# Webots - Setting environment variables
ENV PYTHONPATH ${WEBOTS_HOME}/lib/controller/python:/DWARL
ENV PYTHONIOENCODING $PYTHONIOENCODING:UTF-8

# Install pip and required python packages
RUN apt update
RUN apt upgrade -y
RUN apt install -y python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r /DWARL/requirements.txt

# Jupyter server setup
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

