#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration parameters
IMAGE_NAME=${1:-"pointcloud-processing-sample-cuda"}
IMAGE_TAG=${2:-"latest"}
CONTAINER_NAME=${3:-"cuda-pointcloud-dev"}
WORKSPACE_DIR=${4:-"$(pwd)"}

# Port configuration
VNC_PORT=${VNC_PORT:-5900}
JUPYTER_PORT=${JUPYTER_PORT:-8888}
SSH_PORT=${SSH_PORT:-2222}

# Display configuration information
show_config() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${GREEN}Container Run Configuration${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo -e "Image: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo -e "Container name: ${YELLOW}${CONTAINER_NAME}${NC}"
    echo -e "Workspace directory: ${YELLOW}${WORKSPACE_DIR}${NC}"
    echo -e "VNC port: ${YELLOW}${VNC_PORT}${NC}"
    echo -e "Jupyter port: ${YELLOW}${JUPYTER_PORT}${NC}"
    echo -e "SSH port: ${YELLOW}${SSH_PORT}${NC}"
    echo -e "${BLUE}=================================${NC}\n"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker service is not running${NC}"
        exit 1
    fi
}

# Check if image exists
check_image() {
    if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} >/dev/null 2>&1; then
        echo -e "${RED}Error: Image ${IMAGE_NAME}:${IMAGE_TAG} does not exist${NC}"
        echo -e "Please run the build script first: ${YELLOW}./build.sh${NC}"
        exit 1
    fi
}

# Check and clean up existing container
clean_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Found existing container: ${CONTAINER_NAME}${NC}"
        
        # Check if container is running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${BLUE}Stopping running container...${NC}"
            docker stop ${CONTAINER_NAME}
        fi
        
        echo "Remove old container? [y/N] "
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo -e "${BLUE}Removing old container...${NC}"
            docker rm ${CONTAINER_NAME}
            echo -e "${GREEN}✓ Old container removed${NC}"
        else
            echo -e "${RED}Error: Container name already exists, please choose another name${NC}"
            exit 1
        fi
    fi
}

# Check port occupancy
check_ports() {
    local ports=($VNC_PORT $JUPYTER_PORT $SSH_PORT)
    
    for port in "${ports[@]}"; do
        if lsof -i:$port >/dev/null 2>&1; then
            echo -e "${YELLOW}Warning: Port $port is already in use${NC}"
            echo "Continue? [y/N] "
            read -r response
            if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                echo -e "${RED}Run cancelled${NC}"
                exit 1
            fi
        fi
    done
}

# Display run mode selection
show_mode_selection() {
    echo -e "\n${BLUE}Select run mode:${NC}"
    echo "1) Interactive mode (suitable for development)"
    echo "2) Daemon mode (suitable for long-running processes)"
    echo "3) One-time command mode (execute single command)"
    echo "4) View container status"
    echo "5) Exit"
    echo -n "Please select [1-5]: "
}

# Interactive mode
run_interactive() {
    echo -e "\n${GREEN}Starting interactive container...${NC}"
    
    # Check X11 display
    if [ -n "$DISPLAY" ]; then
        echo -e "${BLUE}Using X11 display: $DISPLAY${NC}"
        X11_VOLUME="-v /tmp/.X11-unix:/tmp/.X11-unix:rw"
        X11_ENV="-e DISPLAY=$DISPLAY"
    else
        echo -e "${YELLOW}Warning: DISPLAY environment variable not set, GUI may not be available${NC}"
        X11_VOLUME=""
        X11_ENV=""
    fi
    
    docker run -it --rm \
        --gpus all \
        --name ${CONTAINER_NAME} \
        --hostname ${CONTAINER_NAME} \
        -p ${VNC_PORT}:5900 \
        -p ${JUPYTER_PORT}:8888 \
        -p ${SSH_PORT}:22 \
        -v ${WORKSPACE_DIR}:/workspace \
        ${X11_VOLUME} \
        ${X11_ENV} \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e VNC_PASSWORD=123456 \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --workdir /workspace \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        /bin/bash
}

# Daemon mode
run_daemon() {
    echo -e "\n${GREEN}Starting daemon container...${NC}"
    
    docker run -d \
        --gpus all \
        --name ${CONTAINER_NAME} \
        --hostname ${CONTAINER_NAME} \
        -p ${VNC_PORT}:5900 \
        -p ${JUPYTER_PORT}:8888 \
        -p ${SSH_PORT}:22 \
        -v ${WORKSPACE_DIR}:/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e VNC_PASSWORD=123456 \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        /startup.sh /bin/bash
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Container started in daemon mode${NC}"
        echo -e "\n${BLUE}Container status:${NC}"
        docker ps --filter "name=${CONTAINER_NAME}"
        
        echo -e "\n${BLUE}View logs:${NC}"
        echo "  docker logs -f ${CONTAINER_NAME}"
        
        echo -e "\n${BLUE}Enter container:${NC}"
        echo "  docker exec -it ${CONTAINER_NAME} /bin/bash"
        
        echo -e "\n${BLUE}Stop container:${NC}"
        echo "  docker stop ${CONTAINER_NAME}"
    else
        echo -e "${RED}✗ Container startup failed${NC}"
    fi
}

# One-time command mode
run_command() {
    echo -e "\n${BLUE}Enter command to execute (e.g., nvidia-smi):${NC}"
    read -r user_command
    
    if [ -z "$user_command" ]; then
        echo -e "${RED}Error: Command cannot be empty${NC}"
        return
    fi
    
    echo -e "\n${GREEN}Executing command: ${user_command}${NC}"
    
    docker run --rm \
        --gpus all \
        -v ${WORKSPACE_DIR}:/workspace \
        -e NVIDIA_VISIBLE_DEVICES=all \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        /bin/bash -c "$user_command"
}

# View container status
show_status() {
    echo -e "\n${BLUE}Container status:${NC}"
    docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
    
    if docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "\n${BLUE}Resource usage:${NC}"
        docker stats --no-stream --filter "name=${CONTAINER_NAME}"
    fi
}

# Display connection information
show_connection_info() {
    echo -e "\n${GREEN}=================================${NC}"
    echo -e "${GREEN}Connection Information${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo -e "${BLUE}VNC:${NC}"
    echo -e "  Address: ${YELLOW}localhost:${VNC_PORT}${NC}"
    echo -e "  Password: ${YELLOW}123456${NC}"
    echo -e "${BLUE}Jupyter:${NC}"
    echo -e "  Address: ${YELLOW}http://localhost:${JUPYTER_PORT}${NC}"
    echo -e "${BLUE}SSH:${NC}"
    echo -e "  Address: ${YELLOW}ssh root@localhost -p ${SSH_PORT}${NC}"
    echo -e "${BLUE}Workspace directory:${NC}"
    echo -e "  Local: ${YELLOW}${WORKSPACE_DIR}${NC}"
    echo -e "  Inside container: ${YELLOW}/workspace${NC}"
    echo -e "${GREEN}=================================${NC}"
}

# Main program
main() {
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}   Docker Run Script${NC}"
    echo -e "${GREEN}=================================${NC}\n"
    
    # Display configuration
    show_config
    
    # Execute checks
    check_docker
    check_image
    clean_container
    
    # Main loop
    while true; do
        show_mode_selection
        read -r mode
        
        case $mode in
            1)
                check_ports
                run_interactive
                show_connection_info
                break
                ;;
            2)
                check_ports
                run_daemon
                show_connection_info
                break
                ;;
            3)
                run_command
                break
                ;;
            4)
                show_status
                break
                ;;
            5)
                echo -e "${YELLOW}Exiting${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid selection, please try again${NC}"
                ;;
        esac
    done
}

# Execute main program
main