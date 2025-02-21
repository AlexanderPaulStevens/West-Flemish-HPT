from beartype import beartype
from dataclasses import dataclass


@beartype
@dataclass
class GPTConfig:
    """GPT Configurations to train the model.

    Args:
        block_size (int): Size of the block.
        vocab_size (int): Size of the vocabulary.
            GPT-2 vocab_size of 50257, padded up to nearest multiple
            of 64 for efficiency
        n_layer (int): Number of layers.
        n_head (int): Number of heads.
        n_embd (int): Embedding size.
        dropout (float): Dropout rate.
        bias (bool): Bias in Linears and LayerNorms, like GPT-2.
            If False, a bit better and faster
        weight_decay (float): Weight decay.
        learning_rate (float): Learning rate.
        betas (tuple): Betas for the optimizer.

    """

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    weight_decay: float = 0.1
    learning_rate: float = 6e-4
    betas: tuple = (0.9, 0.95)
