#!/bin/bash

# Install ngrok if not already installed
if ! command -v ngrok &> /dev/null; then
    echo "Installing ngrok..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
      sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
      echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
      sudo tee /etc/apt/sources.list.d/ngrok.list && \
      sudo apt update && sudo apt install ngrok
fi

# Check if Ollama is running
if ! curl -s localhost:11434 &> /dev/null; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

# Configure ngrok authentication
if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo "Please enter your ngrok authtoken:"
    read -r NGROK_AUTHTOKEN
    ngrok config add-authtoken "$NGROK_AUTHTOKEN"
fi

# Start ngrok tunnel
echo "Starting ngrok tunnel to Ollama..."
ngrok http 11434
