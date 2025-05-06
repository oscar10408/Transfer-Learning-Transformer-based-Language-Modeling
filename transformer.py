

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from transformer.py!")

class MaskedAttention(nn.Module):
    # Masked/Causal Self-Attention
    def __init__(self, embedding_dim, n_head, block_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.attention = nn.Linear(self.embedding_dim, 3*self.embedding_dim) #key, query, and value
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.register_buffer("bias", 
                             torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def split(self, values):
        B, T, C = values.shape
        return values.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    def apply_mask(self, attention):
        T = attention.shape[-1]
        causal_mask = self.bias[:,:,:T,:T]
        attention = attention.masked_fill(causal_mask == 0, float('-inf'))
        return attention

    def forward(self, x):
        Q, K, V  = self.attention(x).split(self.embedding_dim, dim=2)
        Q, K, V = self.split(Q), self.split(K), self.split(V)
        ###########################################################################
        # TODO: Implement the masked attention network.                           #
        # Hint: Please refer to equation(1) in paper - Attention Is All You Need  #
        # link:  https://arxiv.org/pdf/1706.03762.pdf                             #
        #        which shows attention(Q, K, V) = softmax(QK^T/sqrt(d_K))V        #
        #        where d_K is the last dimension of K                             #
        # However, we are implementing masked attention, therefore, before        #
        # softmax, call self.apply_mask(att) to apply causal mask.                #
        # i.e. att = softmax(self.apply_mask(QK^T/sqrt(d_K)))                     #
        # please carefully take transpose and softmax on the correct dimensions.  #
        # Input shapes: Q, K, V (same shapes):                                    #
        #              [batch_size, num_head, sequence_len, embedding_dim/n_head] #
        # Output shape: att:                                                      #
        #              [batch_size, num_head, sequence_len, sequence_len]         #
        ###########################################################################
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        scores = self.apply_mask(scores)
        att = F.softmax(scores, dim=-1)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        y = self.drop1(att) @ V
        y = y.transpose(1, 2).reshape(x.size())
        y = self.drop2(self.fc(y))
        return y

class Block(nn.Module):
    def __init__(self, embedding_dim, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.fc2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.attn = MaskedAttention(embedding_dim, n_head, block_size)
        self.drop = nn.Dropout(0.1)


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        z = nn.GELU()(self.fc1(self.ln2(x)))
        z = x + self.drop(self.fc2(z))
        return z


class Transformer(nn.Module):
    def __init__(self, n_layer, n_head, embedding_dim, vocab_size, block_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = nn.Embedding(block_size, embedding_dim)
        self.drop = nn.Dropout(0.1)
        self.attention = nn.ModuleList([Block(embedding_dim, n_head, block_size) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        _, t = x.size()
        pos = torch.arange(0, t).unsqueeze(0).long().to(x.device)
        tok_emb = self.emb(x)
        pos_emb = self.pos(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.attention:
            x = block(x)
        x = self.ln(x)
        return x


class GPT(nn.Module):
    def __init__(self,
                 n_layer,
                 n_head,
                 embedding_dim,
                 vocab_size,
                 block_size):
        super().__init__()
        self.block_size = block_size
        self.n_layer, self.n_head, self.embedding_dim = n_layer, n_head, embedding_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.transformer = Transformer(self.n_layer,
                                       self.n_head,
                                       self.embedding_dim,
                                       self.vocab_size,
                                       self.block_size)
        
        self.head = nn.Linear(self.embedding_dim,
                              self.vocab_size,
                              bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, inputs, target=None):
        x = self.transformer(inputs)
        logits = self.head(x)
        if target is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            return logits, loss

    def generate(self,
                 inputs,
                 required_chars, 
                 sampling=False,
                 top_k=None):
        with torch.no_grad():
            for _ in range(required_chars): #tokens actually, but the datasets are chars
                logits = self(inputs[:, -self.block_size:])[0][:, -1, :]
                if top_k: #only consider top k candidates
                    top_vals, _ = torch.topk(logits, top_k)
                    logits[logits < top_vals[:, -1]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                if sampling:
                    next_char = torch.multinomial(probs, num_samples=1)
                else:
                    _, next_char = torch.topk(probs, k=1, dim=-1)
                inputs = torch.cat((inputs, next_char), dim=1)
        return inputs