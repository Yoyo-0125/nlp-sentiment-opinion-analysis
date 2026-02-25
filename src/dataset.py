from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, path_or_df, tokenizer, max_len=128):
        # Accept either a file path or a pandas DataFrame
        if isinstance(path_or_df, pd.DataFrame):
            df = path_or_df.copy()
        else:
            df = pd.read_csv(path_or_df)
        df['text'] = df['text'].astype(str)
        df = df.dropna()
        # Ensure integer consecutive indices for __getitem__ using .loc[idx]
        self.data = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, "text"])
        label = int(self.data.loc[idx, "label"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }

def get_dataloader(path, tokenizer, batch_size=16, shuffle=True, max_len=256, num_workers=12):
    dataset = SentimentDataset(path, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def read_dataset(path):
    df = pd.read_csv(path)
    df['text'] = df['text'].astype(str)
    df = df.dropna()
    return df