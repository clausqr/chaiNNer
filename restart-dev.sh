#!/bin/bash

echo "🔄 Restarting chaiNNer development environment..."

echo "📋 Killing all chaiNNer-related processes..."

# Kill all processes related to chaiNNer development
pkill -f "chainner" 2>/dev/null
pkill -f "debugpy" 2>/dev/null
pkill -f "nodemon.*run.py" 2>/dev/null
pkill -f "electron.*chainner" 2>/dev/null
pkill -f "concurrently.*dev:py" 2>/dev/null
pkill -f "python.*run.py" 2>/dev/null
pkill -f "ffmpeg.*v4l2" 2>/dev/null

# Kill processes on specific ports
echo "🔌 Checking for processes on ports 5678 and 8000..."
lsof -ti:5678 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Wait a moment for processes to fully terminate
sleep 2

echo "✅ Cleanup complete. Starting npm run dev..."

# Start the development environment
npm run dev
