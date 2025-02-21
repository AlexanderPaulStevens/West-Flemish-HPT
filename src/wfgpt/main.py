import torch
import pytorch_lightning as pl
from model.hpt import GPT, GPTConfig
import os
from data.dataloader import GPTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize configuration
config = {
    "model_args": {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 384,
        "bias": False,
        "vocab_size": 64,
        "dropout": 0.0,
        "weight_decay": 1e-1,
        "learning_rate": 6e-4,
        "betas": (0.9, 0.99),
    },
    "warmup_iters": 20,
    "lr_decay_iters": 20,
    "min_lr": 1e-3,
    "data_dir": "src/wfgpt/data/",
    "block_size": 64,
    "batch_size": 12,
    "num_workers": 2,
}

# Update model_args with the correct vocab_size
config["model_args"][
    "vocab_size"
] = 65  # based on the meta.pkl file from Karpathy's blog

if __name__ == "__main__":
    # Initialize data and model
    data_module = GPTDataModule(
        config["data_dir"],
        config["block_size"],
        config["batch_size"],
        config["num_workers"],
    )

    model = GPT(GPTConfig(**config["model_args"]))

    outfolder = os.path.join("src/wfgpt/out", "checkpoints")
    # if not os.path.exists(outfolder):
    #    os.makedirs(outfolder)

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=outfolder,  # Directory to save checkpoints
        filename="step_{step}",  # Optional: Custom filename format
        every_n_train_steps=10,  # Save every 10 training steps
        save_top_k=-1,  # Save all checkpoints (set to 1 to keep only the latest)
        monitor=None,  # Do not monitor any metric (save based on steps only)
    )

    trainer = pl.Trainer(
        max_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    model.generate()
