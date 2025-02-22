import pandas as pd
import os
from transformers import AutoTokenizer
import numpy as np

# Read the original CSV file
df = pd.read_csv('src\wfgpt\data\datafolders\Sentences_20211102.csv', encoding='Windows-1252', delimiter=';')

# remove double quotes from the text
text = df['Observation'].str.replace('"', '', regex=False)

# Save the text to a new file
file_path = 'src\wfgpt\data\datafolders\westflemishinput.txt'
text.to_csv(file_path, index=False, header=False)

# Read the file and create a list of sentences (one per line)
with open(file_path, "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file if line.strip()]  # Remove empty lines
print(f"length of dataset in characters: {len(sentences):,}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Save the encodings as .npy files
output_dir = 'src/wfgpt/data/datafolders/encodings'
os.makedirs(output_dir, exist_ok=True)

# Save input_ids, attention_mask, etc.
np.save(os.path.join(output_dir, 'input_ids.npy'), encodings['input_ids'].numpy())
np.save(os.path.join(output_dir, 'attention_mask.npy'), encodings['attention_mask'].numpy())
