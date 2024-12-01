#/bin/bash 

# Build the docker image
DOCKER_IMAGE_NAME=llama_container

echo ""
echo "Building Docker image with nvidia cuda"
echo ""
docker build . -t $DOCKER_IMAGE_NAME

echo ""
echo "Docker image for $DOCKER_IMAGE_NAME has been installed."
echo ""

# Build 
CONTAINER_NAME=llama_container
# Check if the container is running or has exited
if docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    # Prompt user to stop the container
    read -p "Do you want to stop the container (if running) or remove it (if exited)? Type 'y' to stop/remove or any other key to exit: " RESPONSE
    
    if [[ "$RESPONSE" == "y" || "$RESPONSE" == "Y" ]]; then
        # Check if the container is running
        if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
            echo "Stopping container '$CONTAINER_NAME'..."
            docker stop "$CONTAINER_NAME"
            if [[ $? -eq 0 ]]; then
                echo "Container '$CONTAINER_NAME' stopped successfully."
            else
                echo "Failed to stop container '$CONTAINER_NAME'."
            fi
        else
            # Container is not running but may be stopped
            echo "Removing container '$CONTAINER_NAME'..."
            docker rm "$CONTAINER_NAME"
            if [[ $? -eq 0 ]]; then
                echo "Container '$CONTAINER_NAME' removed successfully."
            else
                echo "Failed to remove container '$CONTAINER_NAME'."
            fi
        fi
    else
        echo "Exiting without stopping/removing the container."
        exit
    fi

fi


docker run --name $CONTAINER_NAME --rm --runtime=nvidia --gpus all -it -v $PWD:/workspace $DOCKER_IMAGE_NAME
