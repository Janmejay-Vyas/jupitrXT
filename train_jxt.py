"""
main script to create and train the jupitrXT model
TODO: Update to make a streamlined flow of control for scripts
"""

# Importing the necessary libraries
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken

class CausalSelfAttention(nn.Module):
    """
    Implements a causal self-attention mechanism which is a fundamental component of transformer models
    designed for sequence processing tasks where the model should not have future insight. This module 
    ensures that the predictions for a particular position are dependent only on the known outputs at 
    previous positions.

    Attributes:
        c_attn (nn.Linear): Linear layer that projects input embeddings into queries, keys, and values.
        c_proj (nn.Linear): Linear layer that projects the output of the attention mechanism back to
                            the dimension of embeddings.
        bias (torch.Tensor): Buffer that applies a triangular mask to ensure attention is only applied
                             to preceding positions, preserving causality.
    """

    def __init__(self, config):
        """
        Initializes the CausalSelfAttention layer with specific configuration.

        Args:
            config: A configuration object containing attributes like `n_embd` (embedding size),
                    `n_head` (number of attention heads), and `block_size` (sequence length).
        """
        super().__init__()
        # Ensuring the embedding size is divisible by the number of heads for even split.
        assert config.n_embd % config.n_head == 0

        # Linear transformation that outputs triple the embedding dimension to split into
        # queries, keys, and values.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Linear transformation for the output of the attention computation.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Store the number of attention heads and the embedding dimension per head.
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Register a buffer for the triangular mask that prevents attending to future positions.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                         .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Defines the forward pass of the causal self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: The output tensor after processing with causal self-attention.
        """
        # Unpack the dimensions of the input tensor.
        B, T, C = x.size()

        # Pass the input through the attention projection layer to get combined query, key, value tensors.
        qkv = self.c_attn(x).split(self.n_embd, dim=2)

        # Split and reshape the combined QKV tensor into individual Q, K, V tensors and transpose
        # for multi-head attention computation.
        q, k, v = [tensor.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for tensor in qkv]

        # Compute the attention scores, apply scaling for stability, and use the mask to enforce causality.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Apply softmax to convert scores to probabilities and compute the weighted sum of values.
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        # Project the output back to the embedding dimension and return.
        return self.c_proj(y)


class MLP(nn.Module):
    """
    A multilayer perceptron (MLP) module used within transformer blocks as a position-wise
    feed-forward network. This module is a simple neural network for transforming the 
    representation at every position independently in the sequence.

    Attributes:
        c_fc (nn.Linear): The first linear layer that expands the input dimension.
        gelu (nn.GELU): Gaussian Error Linear Unit (GELU) activation function, which
                        allows the model to include non-linearity and helps in learning
                        more complex patterns. This version uses the 'tanh' approximation
                        for faster computation.
        c_proj (nn.Linear): The second linear layer that projects the output back to 
                            the original embedding dimension.
    """

    def __init__(self, config):
        """
        Initializes the MLP module with specified configurations.

        Args:
            config: A configuration object containing `n_embd`, the size of the input
                    and output embeddings.
        """
        super().__init__()
        # First linear layer that increases dimensionality 4x to allow more complex interactions.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # GELU activation function with 'tanh' approximation.
        self.gelu = nn.GELU(approximate='tanh')
        
        # Second linear layer that reduces dimensionality back to the original size.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        """
        Defines the forward pass of the MLP module.

        Args:
            x (torch.Tensor): The input tensor to the MLP with shape (batch_size, sequence_length, n_embd).

        Returns:
            torch.Tensor: The output tensor after processing through two linear layers
                          and a GELU activation function, with the same shape as input.
        """
        # Pass the input through the first linear layer and then apply the GELU activation function.
        x = self.c_fc(x)
        x = self.gelu(x)
        
        # Finally, pass the activated output through the second linear layer to match the original embedding size.
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Represents a single Transformer block, which is a fundamental component of the Transformer architecture.
    Each block sequentially applies layer normalization, a causal self-attention mechanism, another layer normalization,
    and a multilayer perceptron (MLP). The architecture follows a typical pattern used in JXT models,
    implementing a residual connection around each of the two main sub-layers (self-attention and MLP).

    Attributes:
        ln_1 (nn.LayerNorm): Layer normalization applied before the self-attention mechanism.
        attn (CausalSelfAttention): The causal self-attention module, ensuring that the predictions
                                    for a position are dependent only on the known outputs at previous positions.
        ln_2 (nn.LayerNorm): Layer normalization applied before the MLP.
        mlp (MLP): The multilayer perceptron module that processes the output of the attention mechanism.
    """

    def __init__(self, config):
        """
        Initializes the Transformer block with specified configurations.

        Args:
            config: A configuration object containing necessary parameters like `n_embd`, which is used
                    to set the dimensionality of the layer normalization and to configure the attention and MLP modules.
        """
        super().__init__()
        # Layer normalization that normalizes the embeddings before the self-attention layer.
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        # The self-attention mechanism defined in the CausalSelfAttention class.
        self.attn = CausalSelfAttention(config)
        
        # Layer normalization that normalizes the output of the attention mechanism before passing it to the MLP.
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # The MLP that further processes the output from the attention mechanism.
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Defines the forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor to the block with shape (batch_size, sequence_length, n_embd).

        Returns:
            torch.Tensor: The output tensor from the block, which has the same shape as the input.
                          This output can be fed into subsequent blocks in a Transformer model.
        """
        # Apply layer normalization, then self-attention, and add the result to the input (residual connection).
        x = x + self.attn(self.ln_1(x))
        
        # Apply another layer normalization, then process through the MLP, and add the result to the output
        # of the previous self-attention layer (residual connection).
        x = x + self.mlp(self.ln_2(x))
        
        return x


@dataclass
class JXTConfig:

    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class JXT(nn.Module):
    """
    The JXT class encapsulates the JXT architecture, configuring the model with token and position embeddings,
    multiple transformer blocks, and a final output layer to produce logits over a vocabulary. This class is 
    designed for autoregressive language modeling tasks where each prediction depends only on previous tokens.

    Attributes:
        config (JXTConfig): Configuration object containing model hyperparameters such as the number of layers,
                            embedding dimension, vocabulary size, and maximum sequence length.
        transformer (nn.ModuleDict): Contains the embeddings and transformer blocks.
        lm_head (nn.Linear): Linear transformation applied to the outputs of the transformer blocks to
                             produce logits corresponding to the probability distribution over the vocabulary.
    """

    def __init__(self, config):
        """
        Initializes the JXT model with the specified configuration.

        Args:
            config (JXTConfig): A configuration object specifying model dimensions and architecture parameters.
        """
        super().__init__()
        self.config = config

        # Initializing embeddings and transformer blocks.
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings.
            'wpe': nn.Embedding(config.block_size, config.n_embd),  # Position embeddings.
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks.
            'ln_f': nn.LayerNorm(config.n_embd),  # Final layer normalization.
        })

        # Output layer that projects the final transformer outputs to the vocabulary size.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        """
        Defines the forward pass of the JXT model.

        Args:
            idx (torch.Tensor): Tensor containing token indices of the input sequence (batch_size, sequence_length).

        Returns:
            logits (torch.Tensor): The logits predicting the next token in the sequence
                                            (batch_size, sequence_length, vocab_size).
        """
        B, T = idx.size()
        # Ensure the input does not exceed the configured maximum sequence length.
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Generate position embeddings and add them to the token embeddings.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # Pass the combined embeddings through each transformer block.
        for block in self.transformer.h:
            x = block(x)

        # Apply the final layer normalization.
        x = self.transformer.ln_f(x)

        # Generate logits for each token in the vocabulary.
        logits = self.lm_head(x)

        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"loading weights from pretrained GPT: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized JXT model
        config = JXTConfig(**config_args)
        model = JXT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
# Main program 
# TODO: wrap the main logic in a function

max_length = 30
num_return_sequences = 5

# Auto-detect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# create the prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

model = JXT(JXTConfig())
model.to(device)


# generate predictions from the model
# set the seed to 42
torch.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)