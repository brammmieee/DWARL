#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-u REMOTE_USER] [-h REMOTE_HOST] [-r REMOTE_PATH] [-l LOCAL_PATH]"
    echo "  -u REMOTE_USER   Remote username"
    echo "  -h REMOTE_HOST   Remote hostname"
    echo "  -r REMOTE_PATH   Remote path (e.g., /home/user/DWARL/outputs/)"
    echo "  -l LOCAL_PATH    Local path (e.g., /home/user/DWARL/outputs/)"
    echo "  -i               Interactive mode (prompt for missing arguments)"
    exit 1
}

# Initialize variables
REMOTE_USER=""
REMOTE_HOST=""
REMOTE_PATH=""
LOCAL_PATH=""
INTERACTIVE=false

# Parse command-line arguments
while getopts "u:h:r:l:i" opt; do
    case $opt in
        u) REMOTE_USER="$OPTARG" ;;
        h) REMOTE_HOST="$OPTARG" ;;
        r) REMOTE_PATH="$OPTARG" ;;
        l) LOCAL_PATH="$OPTARG" ;;
        i) INTERACTIVE=true ;;
        *) usage ;;
    esac
done

# Function to prompt for missing information
prompt_if_empty() {
    local var_name="$1"
    local prompt_text="$2"
    if [ -z "${!var_name}" ]; then
        read -p "$prompt_text" $var_name
    fi
}

# If interactive mode or any argument is missing, prompt for input
if $INTERACTIVE || [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PATH" ] || [ -z "$LOCAL_PATH" ]; then
    prompt_if_empty REMOTE_USER "Enter remote username: "
    prompt_if_empty REMOTE_HOST "Enter remote hostname: "
    prompt_if_empty REMOTE_PATH "Enter remote path (e.g., /home/user/DWARL/outputs/): "
    prompt_if_empty LOCAL_PATH "Enter local path (e.g., /home/user/DWARL/outputs/): "
fi

# Ensure paths end with a trailing slash
[[ "${REMOTE_PATH}" != */ ]] && REMOTE_PATH="${REMOTE_PATH}/"
[[ "${LOCAL_PATH}" != */ ]] && LOCAL_PATH="${LOCAL_PATH}/"

# Display the information
echo -e "\nUsing the following information:"
echo "Remote User: ${REMOTE_USER}"
echo "Remote Host: ${REMOTE_HOST}"
echo "Remote Path: ${REMOTE_PATH}"
echo "Local Path: ${LOCAL_PATH}"

# Check if local directory exists
if [ ! -d "${LOCAL_PATH}" ]; then
    echo "Local directory does not exist. Creating it now..."
    mkdir -p "${LOCAL_PATH}"
fi

# Use rsync to copy only non-existent files
echo "Copying new files..."
rsync -av --ignore-existing "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_PATH}"

# Check if the rsync command was successful
if [ $? -eq 0 ]; then
    echo "New files copied successfully!"
else
    echo "Error occurred while copying files."
fi
