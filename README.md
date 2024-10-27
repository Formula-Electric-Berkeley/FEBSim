# Lap Sim

## Overview
--

## Requirements
`requirements.txt`

## How to Build and Run
--


# Build the Docker image - CHANGE DIRECTORY TO WHERE DOCKER FILE IS, for now its Lap Sim
docker build -t molicell-tester .


# Run the Docker container
docker run --rm -v "$(pwd)":/app molicell-tester




