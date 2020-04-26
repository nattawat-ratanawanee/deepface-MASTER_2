#!/bin/bash

docker rm -f deepface-engine-api

docker-compose up -d

docker logs -f deepface-engine-api