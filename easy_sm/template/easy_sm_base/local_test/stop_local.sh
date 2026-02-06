#!/bin/sh

image=$1
port=$2

# Find container running on the specified port with the specified image
container_id=$(docker ps -q --filter "ancestor=${image}" --filter "status=running" | while read cid; do
  container_port=$(docker port "$cid" 2>/dev/null | grep "8080/tcp" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:${port}|${port}" | head -1)
  if [ -n "$container_port" ]; then
    echo "$cid"
    break
  fi
done)

if [ -z "$container_id" ]; then
  echo "No running deployment found for image ${image} on port ${port}"
  exit 0
fi

# Stop the container
docker stop "$container_id"
