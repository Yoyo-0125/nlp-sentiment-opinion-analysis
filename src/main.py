
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from dataset import SentimentDataset, read_dataset, get_dataloader
from models.bert import BERTSentiment
from train import train
from test import test

df = read_dataset("..\data\weibo_senti_100k.csv")
df = df.sample(n=20000, random_state=42)

print("Dataset size:", len(df))

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
test_df, val_df = train_test_split(temp_df, test_size=0.2, random_state=42, stratify=temp_df["label"])

train_loader = get_dataloader(train_df, tokenizer, batch_size=32, shuffle=True)
test_loader = get_dataloader(test_df, tokenizer, batch_size=32, shuffle=False)
val_loader = get_dataloader(val_df, tokenizer, batch_size=32, shuffle=False)

EPOCHS = 3
LR = 2e-5

model = BERTSentiment(num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    train_loss, train_acc, early_break = train(model, train_loader, optimizer, device, epoch=epoch)
    test_loss, test_acc = test(model, val_loader, device)
    print(f"Epoch {epoch+1}/{EPOCHS}: "
        f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
        f"Test loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    if early_break:
        break
