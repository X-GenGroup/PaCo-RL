<h1 align="center"> PaCo-Reward Training Guide </h1>

<div align="center">
  <a href='https://arxiv.org/abs/2512.04784'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://x-gengroup.github.io/HomePage_PaCo-RL/'><img src='https://img.shields.io/badge/ProjectPage-purple?logo=github'></a> &nbsp;
  <a href="https://github.com/X-GenGroup/PaCo-RL"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/collections/X-GenGroup/paco-rl'><img src='https://img.shields.io/badge/Data & Model-green?logo=huggingface'></a> &nbsp;
</div>

## 📋 Table of Contents

- [Get Started](#-get-started)
  - [Environment Setup](#1-environment-setup)
  - [Start Training](#2-start-training)
- [Configuration](#-configuration)
  - [First Token Weight](#first-token-weight)
  - [Batch Size Settings](#batch-size-settings)
- [Hardware Requirements](#-hardware-requirements)
- [Citation](#-citation)



## 🚀 Get Started

### 1. Environment Setup

This project uses a modified version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the training framework.

Clone and install dependencies:
```bash
cd PaCo-Reward
conda create -n paco-reward python=3.12 -y
conda activate paco-reward
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 2. Start Training

Launch training with the provided script:
```bash
cd PaCo-Reward
conda activate paco-reward
bash train/paco_reward.sh
```


### 3. Evaluation

Evaluate reward model on [ConsistencyRank-Bench](https://huggingface.co/datasets/X-GenGroup/ConsistencyRank-Bench) and [EditReward-Bench](https://huggingface.co/datasets/EditScore/EditReward-Bench)

```bash
# Activate an env with vllm installed
conda activate vllm

# Download dataset
hf download X-GenGroup/ConsistencyRank-Bench --repo-type dataset --local-dir ./ConsistencyRank-Bench


# Evaluate on ConsistencyRank
cd Consistency_Rank
python eval_vllm.py \
  --model X-GenGroup/PaCo-Reward-7B \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.85 \
  --batch_size 8

# Print evaluation table
python print_eval_table.py --results_dir results \
  --output_file results_summary.txt
  --output_format txt


# Evaluate on Edit-Bench
cd ../Edit-Bench
bash evaluate_vllm_PaCoReward.sh
```


## ⚙️ Configuration

### First Token Weight

The weight of the first token can be configured in `config/paco_reward_lora.yaml`:
```yaml
first_token_weight: 0.1  # 0.1 is tested the best
```

### Batch Size Settings

The default configuration in `config/paco_reward_lora.yaml` is optimized for 80GB GPUs:
```yaml
per_device_train_batch_size: 8  # Default for 80GB GPUs. Maximum of 16GB with batch_size=1.
```


## 💻 Hardware Requirements

**Minimum Requirements:**
- **GPU Memory**: 16GB (with `per_device_train_batch_size=1`)
- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Training Method**: LoRA fine-tuning

**Recommended Setup:**
- **GPU**: NVIDIA A100 (80GB) or equivalent
- **Batch Size**: 8 per device
- **Training Time**: [about 16h on 2*A100]



## 📚 Model Zoo

PaCo-Reward-7B is built upon Qwen2.5-VL-7B-Instruct and fine-tuned using LoRA for efficient reward modeling in vision-language tasks.

You can find the LoRa adapter [here](X-GenGroup/PaCo-Reward-7B-Lora) and the merged weights [here](X-GenGroup/PaCo-Reward-7B).


## 🤗 Acknowledgement

This training framework is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the authors for their excellent work.

## ⭐ Citation

If you find this work helpful, please cite:
```bibtex
@misc{ping2025pacorladvancingreinforcementlearning,
      title={PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling}, 
      author={Bowen Ping and Chengyou Jia and Minnan Luo and Changliang Xia and Xin Shen and Zhuohang Dang and Hangwei Qian},
      year={2025},
      eprint={2512.04784},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04784}, 
}
```

<div align="center">
  <sub>⭐ Star us on GitHub if you find PaCo-RL helpful!</sub>
</div>