#!/bin/sh

# NOTE: Run this script from the root of the repository (to properly deliver context to the docker image builder).

set -e

# Creates docker image containing the environment for fine-tuning.
docker build \
    --tag juddges_sft:latest \
    --file ./slurm/fine_tuning_env.dockerfile \
    .

# Converts the docker image into an Apptainer image.
docker run \
    --rm \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume $(pwd):/juddges \
    kaczmarj/apptainer build juddges_sft.sif docker-daemon://juddges_sft:latest
