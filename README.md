# FEBSim Docker Setup Instructions

## Overview


## Requirements
`requirements.txt`

## Build the Docker Image
`docker build -t feb-sim-docker .`

## Build the Docker Container
`docker compose up --build`

## Run the Docker Container
`docker run --rm -v "$(pwd)":/app feb-sim-docker`




