import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random

class MLMDataset(Dataset):
    def __init__(self, tokenized_text, vocab, mask_prob=0.15, max_length=128):
        self.tokenized_text = tokenized_text
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.max_length = max_length
        
        # Special Tokens
        self.pad_token = vocab["<PAD>"]
        self.mask_token = vocab["<MASK>"]
        self.cls_token = vocab["<CLS>"]
        self.sep_token = vocab["<SEP>"]

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        tokens = self.tokenized_text[idx].copy()
        labels = tokens.copy()

        # Add [CLS] at the start and [SEP] at the end
        tokens = [self.cls_token] + tokens + [self.sep_token]
        labels = [self.cls_token] + labels + [self.sep_token]

        # Apply 15% masking
        for i in range(1, len(tokens) - 1):  # Avoid masking [CLS] and [SEP]
            if random.random() < self.mask_prob:
                labels[i] = tokens[i]
                tokens[i] = self.mask_token
            else:
                labels[i] = -100

        # Padding
        if len(tokens) < self.max_length:
            padding_needed = self.max_length - len(tokens)
            tokens += [self.pad_token] * padding_needed
            labels += [self.pad_token] * padding_needed
        else:
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]

        return torch.tensor(tokens), torch.tensor(labels)



# Load WikiText-2 dataset
def load_wikitext2(dataset_name = "wikitext-2-v1", sample=1):
    dataset = load_dataset("wikitext",dataset_name, split=f"train[:{int(sample*100)}%]")
    text_data = dataset['text']

    # Initialize vocabulary with special tokens
    vocab = {"<PAD>": 0, "<MASK>": 1, "<CLS>": 2, "<SEP>": 3}
    tokenized_data = []

    for line in text_data:
        if line.strip():
            words = line.strip().split()
            tokenized_line = []
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
                tokenized_line.append(vocab[word])
            tokenized_data.append(tokenized_line)

    return tokenized_data, vocab


if __name__ == '__main__':
    # Load dataset and prepare DataLoader
    tokenized_data, vocab = load_wikitext2()
    dataset = MLMDataset(tokenized_data, vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Example Batch: {next(iter(dataloader))}")
