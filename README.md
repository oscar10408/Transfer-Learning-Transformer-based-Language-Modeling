# Transfer Learning & Transformer-based Language Modeling (PyTorch)

This repository contains two independent deep learning projects implemented in PyTorch:

1. **Transfer Learning for Image Classification** – fine-tuning and freezing ResNet-18 on image datasets.
2. **Transformer-based GPT Model** – a lightweight character-level GPT trained from scratch for sequence modeling.

---

## 📁 Project Structure

.
├── transfer_learning.py # Core training functions for ResNet-based classifier
├── transfer_learning.ipynb # Notebook: training + visualizing classification model
├── transformer.py # Custom Transformer and GPT model implementation
├── transformer_trainer.py # GPT trainer and evaluator utilities
├── transformer.ipynb # Notebook: training and testing character-level GPT

---

## 🧠 Project 1: Transfer Learning for Image Classification

This module demonstrates how to adapt a pretrained ResNet-18 model for binary classification (e.g., ants vs. bees). It includes both full fine-tuning and feature extractor freezing.

### Features:
- Loads `torchvision.models.resnet18` with pretrained weights.
- Modifies only the last classification layer to match custom dataset.
- Supports two training modes:
  - `finetune()`: updates all model layers.
  - `freeze()`: freezes all but the last FC layer.
- Visualizes predictions on validation set.

### Key Functions:
- `train_model(...)`: Epoch-based training + validation.
- `finetune(...)`: Fine-tunes all ResNet layers.
- `freeze(...)`: Freezes feature extractor, trains final classifier.
- `visualize_model(...)`: Displays predictions vs ground truth.

---

## 🔡 Project 2: Transformer-based GPT Language Model

This part of the repo implements a simple character-level GPT using causal attention. It can be used for:
- **Arithmetic learning** (e.g. digit multiplication)
- **Text generation** (e.g. story continuation from prompt)

### Features:
- Manual implementation of Masked (Causal) Self-Attention.
- Multi-head attention, GELU activation, residual connections, and LayerNorm.
- GPT architecture with autoregressive generation.
- Top-k sampling and greedy decoding.
- Supports training on arbitrary character-level datasets.

### Key Classes:
- `MaskedAttention`: Implements causal attention using masking.
- `Block`: Transformer layer with attention + feedforward.
- `Transformer`, `GPT`: Full GPT architecture with token/positional embeddings.
- `Trainer`: Runs training loop, prints loss, and performs sampling.
- `Evaluator`: Computes accuracy or shows text generation results.

---

## ✅ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
