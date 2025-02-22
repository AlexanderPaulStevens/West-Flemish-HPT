import os
import torch
from litgpt.lora import merge_lora_weights
import shutil
import lightning as L

def train_model(model, trainer, init_from, tokenizer=None, data_module=None, config=None):
    if init_from == "scratch":
        if config is None or data_module is None:
            raise ValueError("Config and data_module are required for scratch training")
        trainer.fit(model, datamodule=data_module)
        return model
    elif init_from == "finetuning":
        if tokenizer is None or data_module is None:
            raise ValueError("Tokenizer and data_module are required for fine-tuning")
        finetuned_path = "checkpoints/finetuned_phi2_westflemish.pth"
        
        if os.path.exists(finetuned_path):
            print(f"Loading fine-tuned model from {finetuned_path}...")
            state_dict = torch.load(finetuned_path, map_location="cpu", weights_only=True)
            model.model.load_state_dict(state_dict)
        else:
            print("No fine-tuned model found. Training the model...")
            trainer.fit(model, datamodule=data_module)
            merge_lora_weights(model.model)
            os.makedirs(os.path.dirname(finetuned_path), exist_ok=True)
            torch.save(model.model.state_dict(), finetuned_path)
            print(f"Fine-tuned model saved to {finetuned_path}")
        return model
    else:
        raise ValueError("init_from must be 'scratch' or 'finetuning'")

def get_trainer():
    trainer = L.Trainer(
                devices="auto", accelerator="auto", max_epochs=20,
                max_time='00:00:00:10', accumulate_grad_batches=8,
                callbacks=[PrintCallback()],
            )
    return trainer
      
# Callback for fine-tuning
class PrintCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
        temp_ckpt = "temp_finetuned_cpu.ckpt"
        trainer.save_checkpoint(temp_ckpt, weights_only=True)
        final_ckpt = "checkpoints/finetuned_cpu.ckpt"
        os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
        shutil.move(temp_ckpt, final_ckpt)
        print(f"Checkpoint saved to {final_ckpt}")
