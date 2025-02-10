
#  GPT Implementation from Scratch

  

This repository contains a from-scratch implementation of a GPT (Generative Pre-trained Transformer) model in PyTorch, broken down into educational components.

  

##  Overview

  

The codebase implements core components of the GPT architecture including:

  

- Tokenization

- Embedding layers (token and positional embeddings)

- Multi-head attention mechanism

- Transformer blocks

- Complete GPT model architecture

  

##  Structure

  

The implementation is split across multiple notebooks:

  

###  embedding.ipynb

- Basic text preprocessing and tokenization

- Implementation of a simple tokenizer

- Token and positional embedding implementations

- Basic text generation functionality

  

###  attention.ipynb

- Detailed implementation of attention mechanisms

- Step-by-step breakdown of attention score calculations

- Implementation of both single-head and multi-head attention

- Causal (masked) attention implementation

  

###  implementation.ipynb

- Complete GPT model architecture

- Configuration for GPT-124M model variant

- Model components including:

- Embedding layers

- Transformer blocks

- Layer normalization

- Output head

  

##  Model Configuration

  

The implementation includes configurations for a GPT-124M model with the following specifications:

```python
GPT_CONFIG_124M = {
	"vocab_size": 50257, # Vocabulary size
	"context_length": 1024, # Maximum context length
	"emb_dim": 768, # Embedding dimension
	"n_heads": 12, # Number of attention heads
	"n_layers": 12, # Number of transformer layers
	"drop_rate": 0.1, # Dropout rate
	"qkv_bias": False # Query-Key-Value bias flag
}

```

## Key Components

### Tokenization

- Custom tokenizer implementation with encode/decode functionality

- Support for unknown tokens and end-of-text markers

### Attention Mechanism

- Scaled dot-product attention implementation

- Multi-head attention with parallel processing

- Causal masking for autoregressive generation

### Model Architecture

- Modular implementation of transformer blocks

- Layer normalization

- Residual connections

- Dropout regularization

## Usage

The notebooks can be run sequentially to understand the building blocks of the GPT architecture:

1. Start with `embedding.ipynb` for tokenization and embedding basics

2. Move to `attention.ipynb` to understand attention mechanisms

3. Finally, explore `implementation.ipynb` for the complete model architecture

## Requirements

- PyTorch

- tiktoken (for tokenization)

- Python 3.9+

## Note

This is an educational implementation focused on understanding the core concepts of GPT models.