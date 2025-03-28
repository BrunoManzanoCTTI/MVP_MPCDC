@echo off
REM Set variables
SET REGISTRY=your-registry.com
SET IMAGE_NAME=mpcdc-app
SET TAG=latest

REM Build the Docker image
docker build -t %IMAGE_NAME%:%TAG% .

REM Tag the image for the registry
docker tag %IMAGE_NAME%:%TAG% %REGISTRY%/%IMAGE_NAME%:%TAG%

REM Push the image to the registry
docker push %REGISTRY%/%IMAGE_NAME%:%TAG%

echo Image built and pushed to %REGISTRY%/%IMAGE_NAME%:%TAG%
