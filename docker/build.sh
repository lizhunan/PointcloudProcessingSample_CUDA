#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration parameters
IMAGE_NAME=${1:-"pointcloud-processing-sample-cuda"}
IMAGE_TAG=${2:-"latest"}
DOCKERFILE_PATH=${3:-"."}

# Display configuration information
echo -e "${BLUE}=================================${NC}"
echo -e "${GREEN}Docker Build Configuration${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "Image name: ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "Dockerfile path: ${YELLOW}${DOCKERFILE_PATH}${NC}"
echo -e "${BLUE}=================================${NC}\n"

# Check NVIDIA Docker runtime
check_nvidia_docker() {
    echo -e "${BLUE}Checking NVIDIA Docker runtime...${NC}"
    
    if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected${NC}"
        echo "Please ensure nvidia-docker2 is installed:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
        echo "Continue building? [y/N] "
        read -r response
        if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo -e "${RED}Build cancelled${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ NVIDIA Docker runtime is installed${NC}"
    fi
}

# Check if Docker is running
check_docker_running() {
    echo -e "${BLUE}Checking Docker service...${NC}"
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker service is not running${NC}"
        echo "Please start Docker service:"
        echo "  sudo systemctl start docker"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker service is running${NC}"
}

# Clean up old image
clean_old_image() {
    if docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} >/dev/null 2>&1; then
        echo -e "${YELLOW}Found existing image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
        echo "Remove old image? [y/N] "
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo -e "${BLUE}Removing old image...${NC}"
            docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
            echo -e "${GREEN}✓ Old image removed${NC}"
        else
            echo -e "${YELLOW}Keeping old image, tag will be overwritten${NC}"
        fi
    fi
}

# Display build progress
show_build_info() {
    echo -e "\n${BLUE}Starting Docker image build...${NC}"
    echo "----------------------------------------"
    echo "Build time: $(date)"
    echo "Docker version: $(docker --version)"
    echo "----------------------------------------"
}

# Main build function
build_image() {
    echo -e "\n${BLUE}Executing build...${NC}"
    
    # Set build arguments
    BUILD_ARGS=""
    
    # Add proxy settings if they exist
    if [ ! -z "$HTTP_PROXY" ]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg HTTP_PROXY=$HTTP_PROXY"
    fi
    if [ ! -z "$HTTPS_PROXY" ]; then
        BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$HTTPS_PROXY"
    fi
    
    # Execute docker build
    docker build \
        $BUILD_ARGS \
        --network=host \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -f ${DOCKERFILE_PATH}/Dockerfile \
        ${DOCKERFILE_PATH}
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Image built successfully!${NC}"
        
        # Display image information
        echo -e "\n${BLUE}Image information:${NC}"
        docker images ${IMAGE_NAME}:${IMAGE_TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        # Optional: save build log
        echo -e "\n${BLUE}Build completion time: $(date)${NC}"
        
        return 0
    else
        echo -e "\n${RED}✗ Image build failed!${NC}"
        return 1
    fi
}

# Clean build cache
clean_build_cache() {
    echo -e "\n${BLUE}Checking build cache...${NC}"
    
    # Display current cache size
    CACHE_SIZE=$(docker builder prune -f --filter until=24h --dry-run | grep "Total:" || echo "0B")
    echo -e "Cleanable cache: ${YELLOW}${CACHE_SIZE}${NC}"
    
    if [ "$CACHE_SIZE" != "0B" ]; then
        echo "Clean build cache? [y/N] "
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo -e "${BLUE}Cleaning cache...${NC}"
            docker builder prune -f
            echo -e "${GREEN}✓ Cache cleaned${NC}"
        fi
    fi
}

# Save build information
save_build_info() {
    BUILD_INFO_FILE="build_info_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > ${BUILD_INFO_FILE} << EOF
Build Information
================================
Image name: ${IMAGE_NAME}:${IMAGE_TAG}
Build time: $(date)
Docker version: $(docker --version)
CUDA version: 12.1.0
Base image: nvidia/cuda:12.1.0-devel-ubuntu22.04

Environment variables:
$(env | grep -E "CUDA|NVIDIA|OPEN3D|PANGOLIN")

Build arguments:
$(docker history ${IMAGE_NAME}:${IMAGE_TAG} --no-trunc --format "{{.CreatedBy}}" 2>/dev/null || echo "N/A")
EOF
    
    echo -e "\n${GREEN}✓ Build information saved to: ${BUILD_INFO_FILE}${NC}"
}

# Main program
main() {
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}   Docker Build Script${NC}"
    echo -e "${GREEN}=================================${NC}\n"
    
    # Execute checks
    check_docker_running
    check_nvidia_docker
    
    # Ask about cleaning old image
    clean_old_image
    
    # Display build information
    show_build_info
    
    # Execute build
    if build_image; then
        # Ask about saving build information
        echo ""
        echo "Save build information? [Y/n] "
        read -r response
        if [[ ! "$response" =~ ^([nN][oO]|[nN])$ ]]; then
            save_build_info
        fi
        
        # Ask about cleaning cache
        clean_build_cache
        
        echo -e "\n${GREEN}✓ Build process completed${NC}"
        echo -e "Use the following command to run the container:"
        echo -e "  ${YELLOW}./run.sh${NC}"
        echo -e "or"
        echo -e "  ${YELLOW}./run.sh ${IMAGE_NAME} ${IMAGE_TAG}${NC}"
    else
        echo -e "\n${RED}✗ Build process failed${NC}"
        exit 1
    fi
}

# Execute main program
main