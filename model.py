import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.fc2(self.activation(self.fc1(x))) + x)

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        x = self.ffn(x)
        return x

class GloVeTransformerMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, glove_weights):
        super().__init__()
        # Trainable GloVe Embeddings
        self.embedding = nn.Embedding.from_pretrained(glove_weights, freeze=False)
        self.transformer_layers = nn.Sequential(
            *[TransformerLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # GloVe embeddings
        x = self.transformer_layers(x)  # Pass through transformers
        x = self.output_layer(x)  # Predict masked words
        return F.log_softmax(x, dim=-1)

