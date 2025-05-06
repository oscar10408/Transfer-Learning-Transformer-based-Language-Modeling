# Transfer Learning & Transformer-based Language Modeling (PyTorch)

This repository contains two independent deep learning projects implemented in PyTorch:

1. **Transfer Learning for Image Classification** – fine-tuning and freezing ResNet-18 on image datasets.
2. **Transformer-based GPT Model** – a lightweight character-level GPT trained from scratch for sequence modeling.

---

## 📁 Project Structure
```
.
├── transfer_learning.py # Core training functions for ResNet-based classifier
├── transfer_learning.ipynb # Notebook: training + visualizing classification model
├── transformer.py # Custom Transformer and GPT model implementation
├── transformer_trainer.py # GPT trainer and evaluator utilities
├── transformer.ipynb # Notebook: training and testing character-level GPT
```
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
- `finetune(...)`: Fine-tunes all ResNet layers.
- `freeze(...)`: Freezes feature extractor, trains final classifier.
- `visualize_model(...)`: Displays predictions vs ground truth.
- `train_model(...)`: Epoch-based training + validation.
```python
def train_model(device, dataloaders, dataset_sizes, model, criterion,
                optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # If there is no training happening
    if num_epochs == 0:
        model.eval()
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        best_acc = running_corrects.double() / dataset_sizes['val']

    # Training for num_epochs steps
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss = None
                    preds = None
                    ####################################################################################
                    # TODO: Perform feedforward operation using model, get the labels using            #
                    # torch.max, and compute loss using the criterion function. Store the loss in      #
                    # a variable named loss                                                            #
                    # Inputs:                                                                          #
                    # - inputs : tensor (N x C x H x W)                                                #
                    # - labels : tensor (N)                                                            #
                    # Outputs:                                                                         #
                    # - preds : int tensor (N)                                                         #
                    # - loss : torch scalar                                                            #
                    ####################################################################################
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    ####################################################################################
                    #                             END OF YOUR CODE                                     #
                    ####################################################################################
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model
```

---

## 🔡 Project 2: Transformer-based GPT Language Model

This part of the repo implements a simple character-level GPT using causal attention. It can be used for:
- **Arithmetic learning** (e.g. digit multiplication)
- **Text generation** (e.g. story continuation from prompt)

### 🔧 Core Component: `MaskedAttention`

```python
def forward(self, x):
    Q, K, V = self.attention(x).split(self.embedding_dim, dim=2)
    Q, K, V = self.split(Q), self.split(K), self.split(V)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
    scores = self.apply_mask(scores)
    att = F.softmax(scores, dim=-1)
    y = self.drop1(att) @ V
    return self.drop2(self.fc(y.transpose(1, 2).reshape(x.size())))
```

### 🔄 Model Forward

```python
def forward(self, inputs, target=None):
    x = self.transformer(inputs)
    logits = self.head(x)
    if target is None:
        return logits, None
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    return logits, loss
```

### ✨ Text Generation

```python
def generate(self, inputs, required_chars, top_k=None):
    for _ in range(required_chars):
        logits = self(inputs[:, -self.block_size:])[0][:, -1, :]
        pr F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        inputs = torch.cat((inputs, next_char), dim=1)
    return inputs
```

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
