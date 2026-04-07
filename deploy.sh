#!/bin/bash
# Deployment script for Aesthetic RAG API v2.0
# Usage: ./deploy.sh

set -e

echo "========================================="
echo "Aesthetic RAG API v2.0 - Deployment"
echo "========================================="
echo ""

# Configuration
APP_DIR="/app"
DOCKER_IMAGE="aesthetic-rag-api:v2"
CONTAINER_NAME="aesthetic-rag-api"
PORT=8010

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Check if required files exist
echo "Checking required files..."
if [ ! -f "database.xlsx" ]; then
    echo "ERROR: database.xlsx not found!"
    echo "Please upload database.xlsx to the current directory"
    exit 1
fi

if [ ! -f "api_server.py" ]; then
    echo "ERROR: api_server.py not found!"
    exit 1
fi

echo "✓ Required files found"
echo ""

# Stop and remove old container if exists
echo "Stopping old container (if exists)..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
echo "✓ Old container removed"
echo ""

# Build new Docker image
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE .
echo "✓ Docker image built"
echo ""

# Run new container
echo "Starting new container..."
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p $PORT:8000 \
  -v $(pwd)/database.xlsx:/app/database.xlsx \
  -v $(pwd)/treatment_embeddings.pkl:/app/treatment_embeddings.pkl \
  -e LOCAL_LLM_PROVIDER=transformers \
  -e HF_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  -e HF_MAX_NEW_TOKENS=220 \
  -e TORCH_NUM_THREADS=2 \
  -e MIN_ISSUE_CHARS=5 \
  -e CORS_ALLOW_ORIGINS=* \
  $DOCKER_IMAGE

echo "✓ Container started"
echo ""

# Wait for service to be ready
echo "Waiting for service to start..."
sleep 5

# Test health endpoint
echo "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health || echo "FAILED")

if [[ $HEALTH_RESPONSE == *"ok"* ]]; then
    echo "✓ Service is healthy!"
else
    echo "✗ Service health check failed!"
    echo "Response: $HEALTH_RESPONSE"
    echo ""
    echo "Checking logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Deployment successful!"
echo "========================================="
echo ""
echo "Service Information:"
echo "  - Container: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - Image: $DOCKER_IMAGE"
echo ""
echo "API Endpoints:"
echo "  - Health: http://143.198.238.226:$PORT/health"
echo "  - Sub-zones: POST http://143.198.238.226:$PORT/subzones"
echo "  - Search: POST http://143.198.238.226:$PORT/search"
echo ""
echo "Useful Commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop service: docker stop $CONTAINER_NAME"
echo "  - Start service: docker start $CONTAINER_NAME"
echo "  - Restart service: docker restart $CONTAINER_NAME"
echo "  - Remove service: docker rm -f $CONTAINER_NAME"
echo ""
echo "Testing the API:"
echo "  curl -X POST http://143.198.238.226:$PORT/subzones \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"region\": \"Face\"}'"
echo ""
