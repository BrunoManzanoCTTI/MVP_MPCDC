#!/bin/bash

# Set variables
REGISTRY="your-registry.com"  # Replace with your actual registry
IMAGE_NAME="mpcdc-app"
TAG="latest"

# Build the Docker image
docker build -t $IMAGE_NAME:$TAG .

# Tag the image for the registry
docker tag $IMAGE_NAME:$TAG $REGISTRY/$IMAGE_NAME:$TAG

# Push the image to the registry
docker push $REGISTRY/$IMAGE_NAME:$TAG

echo "Image built and pushed to $REGISTRY/$IMAGE_NAME:$TAG"
