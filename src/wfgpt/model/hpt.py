import torch
from torch import nn
from torch.nn import functional as F

import math

from model.base import LayerNorm, Block

from beartype import beartype
from beartype import typing as tp
from jaxtyping import Num, Int
import os
import logging

import lightning as L
import litgpt
from litgpt.lora import GPT as LoRAGPT
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

@beartype
class ScratchGPT(L.LightningModule):
    def __init__(self, model_params: dict):
        """GPT model, based on Andrej Karpathy's nanoGPT with Pytorch Lightning.

        Args:
            model_params: Parameters to train the model.
        """
        super().__init__()
        self.model_params = model_params

        # init model components
        self.transformer = nn.ModuleDict(
            dict(

                wte=nn.Embedding(model_params['vocab_size'], model_params['n_embd']),
                wpe=nn.Embedding(model_params['block_size'], model_params['n_embd']),
                drop=nn.Dropout(model_params['dropout']),
                h=nn.ModuleList([Block(model_params) for _ in range(model_params['n_layer'])]),
                ln_f=LayerNorm(model_params['n_embd'], bias=model_params['bias']),
            )
        )
        self.lm_head = nn.Linear(model_params['n_embd'], model_params['vocab_size'], bias=False)

        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(

                    p, mean=0.0, std=0.02 / math.sqrt(2 * model_params['n_layer'])

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

        assert t <= self.model_params['block_size'], ValueError(
            f"Cannot forward sequence of length {t}, block size is only {self.model_params['block_size']}"

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

            {"params": decay_params, "weight_decay": self.model_params['weight_decay']},

            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,

            lr=self.model_params['learning_rate'],
            betas=self.model_params['betas'],

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

                if idx.size(1) <= self.model_params['block_size']
                else idx[:, -self.model_params['block_size'] :]

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

class LitLLM(L.LightningModule):
    def __init__(self, tokenizer: tp.Any):
        """Lightning Module for fine-tuning a LoRA-augmented GPT model.

        Args:
            tokenizer: Tokenizer compatible with the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.model = LoRAGPT.from_name(
            name="phi-2",
            lora_r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_query=True,
            lora_key=False,
            lora_value=True,
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)
        self.save_hyperparameters(ignore=['tokenizer'])

    def setup(self, stage= None) -> None:
        """Setup method to load model weights or initialize from pretrained.

        Args:
            stage: Optional training stage identifier
        """
        checkpoint_path = "checkpoints/microsoft/phi-2/lit_model.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("No checkpoint found. Loading pretrained Microsoft phi-2 model...")
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2", torch_dtype=torch.float32
            )
            self.model.load_state_dict(pretrained_model.state_dict(), strict=False)

    def training_step(
        self, 
        batch: tp.Dict[str, Int[torch.Tensor, "N D"]], 
    ) -> Num[torch.Tensor, ""]:
        """Training step for the model.

        Args:
            batch: Dictionary containing input_ids and labels
            batch_idx: Index of the current batch

        Returns:
            Loss tensor
        """
        input_ids, targets = batch["input_ids"], batch["labels"]

        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(
            logits[..., :-1, :], 
            targets[..., 1:], 
            ignore_index=self.tokenizer.pad_token_id
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self) -> tp.Tuple[
        tp.List[torch.optim.Optimizer],
        tp.List[tp.Dict[str, tp.Any]]
    ]:
        """Configure optimizers and schedulers.

        Returns:
            Tuple of optimizer list and scheduler config list
        """
        warmup_steps = 10
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.0002, 
            weight_decay=0.0, 
            betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda step: min(1.0, float(step) / warmup_steps)
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 50
    ) -> str:
        """Generate text based on a prompt.

        Args:
            prompt: Input prompt string
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of top probabilities to consider

        Returns:
            Generated text string
        """
        self.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            logits = self.model(generated_ids)
            next_token_logits = logits[:, -1, :] / temperature
            top_k_logits, top_k_indices = next_token_logits.topk(top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token_idx)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
