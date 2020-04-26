#!/bin/bash

docker rmi -f deepface-engine-image:1.0

docker build -t deepface-engine-image:1.0 .

# docker save -o deepface-engine-image.tar deepface-engine-image:1.0
# docker load -i .\\deepface-engine-image.tar