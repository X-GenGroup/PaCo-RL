# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

benchmark_dir="EditScore/EditReward-Bench"

model_name="PaCo-Reward-7B"
model_path="Jayce-Ping/PaCo-Reward-7B"
python evaluation.py \
    --benchmark_dir $benchmark_dir \
    --scorer ConsistencyScore \
    --result_dir results/$model_name \
    --backbone qwen25vl_vllm \
    --model_name_or_path $model_path \
    --score_range 25 \
    --max_workers 1 \
    --max_model_len 4096 \
    --max_num_seqs 1 \
    --max_num_batched_tokens 4096 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.5 \
    --num_pass 1

python calculate_statistics.py \
    --result_dir results/EditScore-7B/qwen25vl