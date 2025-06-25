#!/bin/bash

# Setup script for NeuroRiskLogic deployment

echo "Setting up NeuroRiskLogic deployment..."

# Check if .env exists, if not copy from template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your actual values before proceeding."
    echo "Press any key to continue after editing..."
    read -n 1 -s
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p ../data/models ../data/synthetic ../logs ../reports

# Check if model file exists
if [ ! -f ../data/models/model_current.pkl ]; then
    echo "WARNING: No trained model found at ../data/models/model_current.pkl"
    echo "You need to train a model first using: python scripts/train_model.py"
fi

# Build images
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check health
echo "Checking service health..."
curl -f http://localhost:8000/health || echo "API not yet ready"

echo "Deployment setup complete!"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f api"
echo "  docker-compose logs -f retraining"
echo ""
echo "To stop services:"
echo "  docker-compose down"