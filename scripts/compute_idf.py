"""
Compute per-token IDF (inverse document frequency) weights over the downloaded
pretraining shards, used by DeepSeekEngram to decide which tokens are "rare"
vs "common". Run once after tokenizer training and dataset download, before
base_train.py (or before constructing any model that calls engram.load_idf()).

Usage:
    python -m scripts.compute_idf
"""
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from nanochat.tokenizer import get_tokenizer

print("Loading trained nanochat tokenizer...")
tokenizer = get_tokenizer()
vocab_size = 32768

base_cache_dir = "/root/.cache/nanochat"
data_dir = os.path.join(base_cache_dir, "base_data_climbmix")

if not os.path.exists(data_dir):
    for item in os.listdir(base_cache_dir):
        full_path = os.path.join(base_cache_dir, item)
        if os.path.isdir(full_path) and any(f.endswith('.parquet') for f in os.listdir(full_path)):
            data_dir = full_path
            break

parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"Could not find any dataset parquet shards under {base_cache_dir}")

print(f"Found {len(parquet_files)} data shards. Extracting text documents...")

docs = []
for f in parquet_files:
    df_shard = pd.read_parquet(f)
    col = 'text' if 'text' in df_shard.columns else ('content' if 'content' in df_shard.columns else df_shard.columns[0])
    docs.extend(df_shard[col].tolist())

# 1.77M docs is too heavy for raw CPU loops -- sample down to a size that keeps
# the same normalized frequency distribution shape at much lower compute cost.
SAMPLE_SIZE = 50000
if len(docs) > SAMPLE_SIZE:
    print(f"Downsampling dataset from {len(docs)} to {SAMPLE_SIZE} random documents for speed...")
    random.seed(42)
    docs = random.sample(docs, SAMPLE_SIZE)

print(f"Calculating token IDF from {len(docs)} documents...")
df = torch.zeros(vocab_size, dtype=torch.float32)

for i, doc in enumerate(docs):
    if i % 10000 == 0 and i > 0:
        print(f"  -> Processed {i}/{len(docs)} documents...")

    tokens = np.array(tokenizer.encode(doc))
    unique_tokens = np.unique(tokens)
    unique_tokens = unique_tokens[unique_tokens < vocab_size]
    df[unique_tokens] += 1.0

num_docs = len(docs)
idf = torch.log(torch.tensor(num_docs) / (1.0 + df))
idf_normalized = (idf - idf.min()) / (idf.max() - idf.min() + 1e-8)

torch.save(idf_normalized, "idf_weights.pt")
print("idf_weights.pt successfully saved to workspace root!")
