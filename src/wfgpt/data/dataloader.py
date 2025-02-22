import torch
import numpy as np
import lightning as L
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class GPTDataset(Dataset):
    def __init__(self, input_ids, block_size):
        """
        Args:
            input_ids (np.ndarray): Loaded input_ids from .npy file, shape (num_samples, seq_length).
            block_size (int): Maximum sequence length for training.
        """
        self.input_ids = input_ids
        self.block_size = block_size

    def __len__(self):
        # Number of samples depends on how many full block_size sequences we can extract
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns a single sample with input (x) and target (y) shifted by one token.
        """
        sequence = self.input_ids[idx]
        # Truncate or pad to block_size
        if len(sequence) > self.block_size:
            sequence = sequence[:self.block_size]
        else:
            sequence = np.pad(sequence, (0, self.block_size - len(sequence)), mode='constant', constant_values=0)
        
        x = torch.tensor(sequence[:-1], dtype=torch.long)  # Input: all but last token
        y = torch.tensor(sequence[1:], dtype=torch.long)   # Target: all but first token
        return x, y
    
class GPTDataModule(L.LightningDataModule):
    def __init__(self, data_dir, block_size, batch_size, num_workers, encodings_dir='src/wfgpt/data/datafolders/encodings'):
        """
        Args:
            data_dir (str): Base directory for data (kept for compatibility).
            block_size (int): Maximum sequence length.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            encodings_dir (str): Directory where .npy encodings are stored.
        """
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encodings_dir = encodings_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def setup(self, stage=None):
        """
        Loads the BERT-tokenized input_ids.npy and splits into train/val datasets.
        """
        input_ids_path = os.path.join(self.encodings_dir, 'input_ids.npy')
        if not os.path.exists(input_ids_path):
            raise FileNotFoundError(f"File not found: {input_ids_path}")
        
        # Load input_ids
        input_ids = np.load(input_ids_path)
        print(f"Loaded input_ids with shape: {input_ids.shape}")

        # Simple train/val split (e.g., 90% train, 10% val)
        split_idx = int(0.9 * len(input_ids))
        train_data = input_ids[:split_idx]
        val_data = input_ids[split_idx:]

        # Create datasets
        self.train_dataset = GPTDataset(train_data, self.block_size)
        self.val_dataset = GPTDataset(val_data, self.block_size)
        print(f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

class WestFlemishData(Dataset):
    def __init__(self, encodings_dir='src/wfgpt/data/datafolders/encodings'):
        import numpy as np
        self.encodings_dir = encodings_dir
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.load_encodings()
        
    def load_encodings(self):
        import numpy as np
        input_ids_path = os.path.join(self.encodings_dir, 'input_ids.npy')
        if not os.path.exists(input_ids_path):
            raise FileNotFoundError(f"File not found: {input_ids_path}")
        self.input_ids = np.load(input_ids_path)
        print(f"Encodings loaded successfully. Total samples: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)
    
    def create_collate_fn(self, pad_token_id):
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            labels = [item["labels"] for item in batch]
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
            return {"input_ids": input_ids_padded, "labels": labels_padded}
        return collate_fn

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.int64)
        labels = input_ids.clone()
        dataset = {"input_ids": input_ids, "labels": labels}
        collate_fn = self.create_collate_fn(dataset.pad_token_id)
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=True, collate_fn=collate_fn,
            num_workers=0, pin_memory=True
        )
        return dataloader
