import os
import sys
sys.dont_write_bytecode = True
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from model.sampler import sample_from_model
from model.hpt import ScratchGPT
from transformers import AutoTokenizer
from model.hpt import LitLLM
from model.config import GPTConfig
from data.dataloader import GPTDataModule, WestFlemishData
import numpy as np
from model.trainer import train_model, get_trainer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # Set seed for reproducibility
    L.seed_everything(42)
    init_from = "scratch"  # Options: "scratch" or "finetuning"

    if init_from == "scratch":
        # Load BERT tokenizer and set vocab_size
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = GPTConfig()
        config.vocab_size = tokenizer.vocab_size  # ~30,522

        # Determine block_size from encodings before initializing data_module
        input_ids = np.load(config.encodings_path)
        config.block_size = input_ids.shape[1]

        # Setup data module with BERT encodings
        data_module = GPTDataModule(
            data_dir=config.data_dir,
            block_size=config.block_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            encodings_dir=config.encodings_dir
        )
        data_module.setup()  # Load encodings

        outfolder = os.path.join("src/wfgpt/out", "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=outfolder, filename="step_{step}", every_n_train_steps=10,
            save_top_k=-1, monitor=None
        )
        trainer = get_trainer()

        architecture = ScratchGPT(config.model_args)

        model = train_model(architecture, trainer, init_from, config=config, data_module=data_module)

    elif init_from == "finetuning":
        # Setup for fine-tuning
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataloader = WestFlemishData()
        
        trainer = get_trainer()

        architecture = LitLLM(tokenizer=tokenizer)

        model = train_model(architecture, trainer, init_from, tokenizer=tokenizer, dataloader=dataloader)

    sample_from_model(model, init_from, tokenizer=tokenizer)

    