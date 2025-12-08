<h1 align="center"> PaCo-RL </h1>

<p align="center">
  <b>Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling</b>
</p>

<div align="center">
  <a href='https://arxiv.org/abs/2512.04784'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://x-gengroup.github.io/HomePage_PaCo-RL/'><img src='https://img.shields.io/badge/ProjectPage-purple?logo=github'></a> &nbsp;
  <a href="https://github.com/X-GenGroup/PaCo-RL"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/collections/X-GenGroup/paco-rl'><img src='https://img.shields.io/badge/Data & Model-green?logo=huggingface'></a> &nbsp;
</div>

## 🌟 Overview

**PaCo-RL** is a comprehensive framework for consistent image generation through reinforcement learning, addressing challenges in preserving identities, styles, and logical coherence across multiple images for storytelling and character design applications.

### Key Components

- **PaCo-Reward**: A pairwise consistency evaluator with task-aware instruction and CoT reasoning.
- **PaCo-GRPO**: Efficient RL optimization with resolution-decoupled training and log-tamed multi-reward aggregation

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/X-GenGroup/PaCo-RL.git
cd PaCo-RL
```

### Train Reward Model
```bash
cd PaCo-Reward
conda create -n paco-reward python=3.12 -y
conda activate paco-reward
cd LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation
cd .. && bash train/paco_reward.sh
```

See 📖 [PaCo-Reward Documentation](PaCo-Reward/README.md) for detailed guide.

### Run RL Training
```bash
cd PaCo-GRPO
conda create -n paco-grpo python=3.12 -y
conda activate paco-grpo
pip install -e .

# Setup vLLM reward server
conda create -n vllm python=3.12 -y
conda activate vllm && pip install vllm
export CUDA_VISIBLE_DEVICES=0
export VLLM_MODEL_PATHS='X-GenGroup/PaCo-Reward-7B'
export VLLM_MODEL_NAMES='Paco-Reward-7B'
bash vllm_server/launch.sh

# Start training
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
conda activate paco-grpo
bash scripts/single_node/train_flux.sh t2is
```

See 📖 [PaCo-GRPO Documentation](PaCo-GRPO/README.md) for detailed guide.

## 📁 Repository Structure
```
PaCo-RL/
├── PaCo-GRPO/              # RL training framework
│   ├── config/             # RL configurations
│   ├── scripts/            # Training scripts
│   └── README.md
├── PaCo-Reward/            # Reward model training
│   ├── LLaMA-Factory/      # Training framework
│   ├── config/             # Training configurations
│   └── README.md
└── README.md
```

## 🎁 Model Zoo

| Model | Type | HuggingFace |
|-------|------|-------------|
| **PaCo-Reward-7B** | Reward Model | [🤗 Link](https://huggingface.co/X-GenGroup/PaCo-Reward-7B) |
| **PaCo-Reward-7B-Lora** | Reward Model (LoRA) | [🤗 Link](https://huggingface.co/X-GenGroup/PaCo-Reward-7B-Lora) |
| **PaCo-FLUX.1-dev** | T2I Model (LoRA) | [🤗 Link](https://huggingface.co/X-GenGroup/PaCo-FLUX.1-dev-Lora) |
| **PaCo-FLUX.1-Kontext-dev** | Image Editing Model (LoRA) | [🤗 Link](https://huggingface.co/X-GenGroup/PaCo-FLUX.1-Kontext-Lora) |
| **PaCo-QwenImage-Edit** | Image Editing Model (LoRA) | [🤗 Link](https://huggingface.co/X-GenGroup/PaCo-Qwen-Image-Edit-Lora) |

## 🤗 Acknowledgement

Our work is built upon [Flow-GRPO](https://github.com/yifan123/flow_grpo), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [vLLM](https://github.com/vllm-project/vllm), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL). We sincerely thank the authors for their valuable contributions to the community.

## ⭐ Citation
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

## 📬 Contact

If you have any inquiries, suggestions, or wish to contact us for any reason, we warmly invite you to email us at  jayceping6@gmail.com or cp3jia@stu.xjtu.edu.cn.

<div align="center">
  <sub>⭐ Star us on GitHub if you find PaCo-RL helpful!</sub>
</div>

