from beartype import beartype
from dataclasses import dataclass


@beartype
@dataclass

class GPTConfig():
    def __init__(self):
        """GPT Configurations to train the model.

        Args:
        block_size (int): Size of the block. Default is 1024.
        vocab_size (int): Size of the vocabulary. Default is 50304.
        n_layer (int): Number of layers. Default is 12.
        n_head (int): Number of heads. Default is 12.
        n_embd (int): Embedding size. Default is 768.
        dropout (float): Dropout rate. Default is 0.0.
        bias (bool): Bias in Linears and LayerNorms, like GPT-2. Default is True.
        weight_decay (float): Weight decay. Default is 0.1.
        learning_rate (float): Learning rate. Default is 6e-4.
        betas (tuple): Betas for the optimizer. Default is (0.9, 0.95).
        warmup_iters (int): Number of warmup iterations. Default is 20.
        lr_decay_iters (int): Number of learning rate decay iterations. Default is 20.
        min_lr (float): Minimum learning rate. Default is 1e-3.
        batch_size (int): Batch size. Default is 12.
        num_workers (int): Number of workers. Default is 2.
        encodings_dir (str): Directory for encodings. Default is 'src/wfgpt/data/datafolders/encodings'.
        encodings_path (str): Path for encodings. Default is 'src/wfgpt/data/datafolders/encodings/input_ids.npy'.
        data_dir (str): Directory for data. Default is 'src/wfgpt/data/'.
        model_args (dict): Dictionary of model arguments including n_layer, n_head, n_embd, bias, vocab_size, dropout, weight_decay, learning_rate, betas, and block_size.
        """
        super().__init__()
        self.block_size: int = 1024
        self.vocab_size: int = 50304
        self.n_layer: int = 12
        self.n_head: int = 12
        self.n_embd: int = 768
        self.dropout: float = 0.0
        self.bias: bool = True
        self.weight_decay: float = 0.1
        self.learning_rate: float = 6e-4
        self.betas: tuple = (0.9, 0.95)
        self.warmup_iters: int =  20
        self.lr_decay_iters: int= 20
        self.min_lr: float = 1e-3
        self.batch_size : int= 12
        self.num_workers: int = 2
        self.encodings_dir: str = 'src/wfgpt/data/datafolders/encodings'
        self.encodings_path: str = 'src/wfgpt/data/datafolders/encodings/input_ids.npy'
        self.data_dir: str = 'src/wfgpt/data/'
        
        self.model_args = {
                'n_layer': self.n_layer, 'n_head': self.n_head, 'n_embd': self.n_embd,
                'bias': self.bias, 'vocab_size': self.vocab_size,
                'dropout': self.dropout, 'weight_decay': self.weight_decay,
                'learning_rate': self.learning_rate, 'betas': self.betas,
                'block_size': self.block_size
            }