#!/bin/bash

# Create new tmux session named "infer-web"
tmux new-session -d -s infer-web

# Add ngrok authtoken
tmux send-keys -t infer-web 'ngrok config add-authtoken 2Nsu4GJYzUOggvogEc4f2KSJ5EI_5S7t69AdyERBRxkns72ru' C-m

# Run `ngrok http 8007` in the "infer-web" tmux session
tmux send-keys -t infer-web 'ngrok http 8007' C-m

# Create a new vertical pane in the "infer-web" tmux session
tmux split-window -v -t infer-web

# Source the virtual environment
tmux send-keys -t infer-web.1 'source .venv/bin/activate' C-m

# Run `uvicorn main:app --host 0.0.0.0 --port 8007` in the new pane of the "infer-web" tmux session
tmux send-keys -t infer-web.1 'uvicorn main:app --host 0.0.0.0 --port 8007' C-m

# Attach to the "infer-web" tmux session
tmux attach-session -t infer-web