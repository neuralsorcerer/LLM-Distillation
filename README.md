# LLM Knowledge Distillation & Simplified PPO using Reinforcement Learning

This repository demonstrates how to:

1. Distill a `gpt2-medium` teacher into a smaller `gpt2` student model.
2. Further align the student with a **simplified PPO** loop using a basic reward signal.

## Overview
- **Teacher**: `gpt2-medium`
- **Student**: `gpt2`
- **Dataset**: WikiText-2 (filtered for demonstration)
- **Distillation**: Combines cross-entropy and KL-divergence losses with temperature scaling.
- **PPO Fine-Tuning**: Very simplified approach that uses cosine similarity of token embeddings as a reward signal.


## Installation
### Clone the Repository:
```bash
git clone https://github.com/neuralsorcerer/LLM-Distillation.git
cd LLM-Distillation
```

### Create and Activate a Virtual Environment (Optional but recommended):
```bash
conda create -n distillation_env python=3.8 -y
conda activate distillation_env
```
*(Alternatively, use `pipenv` or a standard `venv`)*

### Install Dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets matplotlib numpy tqdm
```

## Usage
1. Open the notebook (`LLMdistillation.ipynb`) in Jupyter or Colab.
2. Run each cell in order:
    - The data is automatically downloaded from the WikiText-2 dataset.
    - Distillation is performed with the `DistillationTrainer` class.
    - A small demonstration of PPO fine-tuning is run with the `PPOTrainerCustom` class.
3. Check logs and outputs:
    - Perplexities for teacher vs. student.
    - Loss curves for both distillation and PPO phases.

## Tips & Notes
- If you run out of GPU memory, reduce `BATCH_SIZE` or use smaller model variants (e.g., `distilgpt2` as a teacher).
- The PPO step here is heavily simplified—use a more complete framework if you need proper policy gradients, advantage estimation, clipping, etc.
- Adjust **hyperparameters** (learning rate, temperature, etc.) in the `Config` class to experiment.

## License
This project is provided under the [MIT License](LICENSE).

## References
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Knowledge Distillation (Hinton et al.)](https://arxiv.org/abs/1503.02531)
- [Proximal Policy Optimization (Schulman et al.)](https://arxiv.org/abs/1707.06347)
- [WikiText-2](https://paperswithcode.com/dataset/wikitext-2)
