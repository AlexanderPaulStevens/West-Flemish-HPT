import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

import math

from model.base import LayerNorm, Block
from model.config import GPTConfig
from beartype import beartype
from beartype import typing as tp
from jaxtyping import Num, Int
import numpy as np
import logging

logger = logging.getLogger(__name__)


@beartype
class GPT(pl.LightningModule):
    def __init__(self, config: GPTConfig):
        """GPT model, based on Andrej Karpathy's nanoGPT with Pytorch Lightning.

        Args:
            config (GPTConfig): Configuration to train the model.
        """
        super().__init__()
        self.config = config

        # init model components
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        logger.info("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Gets the number of parameters of the model.

        Args:
            non_embedding (bool): If True (default), the number of parameters of the embeddings is not considered.

        Returns:
            int: Number of parameters of the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= int(self.transformer.wpe.weight.numel())
        return n_params

    def _init_weights(self, module: nn.Module):
        """Initialize the weights of the model.

        Args:
            module (nn.Module): Module for which the weights should be initialized.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Int[torch.Tensor, "N D"],
        targets: Num[torch.Tensor, "N D"] | None = None,
    ) -> Num[torch.Tensor, "N D"]:
        """Forward pass of the model, required by Pytorch Lightning.

        Args:
            idx (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Output tensor as logits.
        """
        _, t = idx.size()
        assert t <= self.config.block_size, ValueError(
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim

        return logits

    def training_step(
        self, batch: list[Num[torch.Tensor, "N D"]]
    ) -> tp.Dict[str, Num[torch.Tensor, ""]]:
        """Training step of the model, required by Pytorch Lightning.

        Args:
            batch (list): Batch of data.

        Returns:
            Dict[str, torch.Tensor]: Loss of the model for given batch.
        """

        idx, targets = batch
        logits = self(idx, targets)

        loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Defines optimizer and scheduler for the model, required by Pytorch Lightning."""

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Num[torch.Tensor, "N D"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
