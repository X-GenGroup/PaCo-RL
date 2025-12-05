accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=2 --main_process_port 29501 dist_test.py