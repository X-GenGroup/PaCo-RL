CONFIG=$1
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Audo-set number of GPUs if not set
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Count number of GPUs from CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=$NUM_GPUS \
    --main_process_port $MAIN_PROCESS_PORT \
    scripts/train_flux_kontext.py \
    --config config/grpo.py:$CONFIG