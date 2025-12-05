#!/bin/bash
# Stop all services
# Usage: bash stop.sh

VLLM_LABEL=${VLLM_LABEL:-"vllm"}
VLLM_PORT=${VLLM_PORT:-8000}
OUTPUT_DIR="${VLLM_LABEL}"

echo "🛑 Stopping all services..."

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "⚠️  No output directory found: $OUTPUT_DIR"
    exit 0
fi

# Stop vLLM servers
for PID_FILE in "$OUTPUT_DIR"/*.pid; do
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        MODEL_NAME=$(basename "$PID_FILE" .pid)
        echo -n "   $MODEL_NAME (PID $PID)... "
        
        if kill -0 $PID 2>/dev/null; then
            kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null
            echo "✅"
        else
            echo "⚠️  Not running"
        fi
    fi
done

# Stop gateway (tmux session)
if [ -f "$OUTPUT_DIR/gateway.tmux" ]; then
    TMUX_SESSION=$(cat "$OUTPUT_DIR/gateway.tmux")
    echo -n "   Gateway (tmux: $TMUX_SESSION)... "
    
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
        echo "✅"
    else
        echo "⚠️  Not running"
    fi
fi

# Force kill gateway by port
GATEWAY_PID=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
if [ -n "$GATEWAY_PID" ]; then
    echo -n "   Gateway (port $VLLM_PORT)... "
    kill -9 $GATEWAY_PID 2>/dev/null
    echo "✅"
fi

# Remove entire output directory
rm -rf "$OUTPUT_DIR"

echo "✅ All services stopped and cleaned up"