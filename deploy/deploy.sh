#!/bin/bash
echo "Pulling latest images and restarting containers..."
docker compose up -d --pull always --force-recreate
echo "Done."
