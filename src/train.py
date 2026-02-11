import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

from dataset import SentimentDataset, read_dataset, get_dataloader
from models.bert import BERTClassifier
from models.lstm import LSTMClassifier


def train(model, train_loader, optimizer, device, num_classes, epoch):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    early_stop = False

    for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}]", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask)
        if num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
        else:
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.1)

        loss.backward()
        optimizer.step()

        if num_classes == 1:
                predicted = (outputs > 0).long()
        else:
            _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        if loss.item() < 0.001:
            print("\nEarly Stopped: ", loss.item())
            early_stop = True
            break

    # 从环境变量读取保存路径
    model_save_path = os.getenv('MODEL_SAVE_PATH', './src/models/lstm_sentiment_small.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc, early_stop


def evaluate(model, test_loader, device, num_classes=1):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            if num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
            else:
                loss = F.cross_entropy(outputs, labels)

            if num_classes == 1:
                predicted = (outputs > 0).long()
            else:
                _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    epoch_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    df = read_dataset('..\\data\\weibo_senti_100k.csv')
    # df = df.sample(n=10000, random_state=42)

    print('Dataset size:', len(df))

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    test_df, val_df = train_test_split(temp_df, test_size=0.2, random_state=42, stratify=temp_df['label'])

    # 从环境变量读取 tokenizer 路径
    tokenizer_path = os.getenv('TOKENIZER_PATH', './src/models/chinese-roberta-wwm-ext')
    model_save_path = os.getenv('MODEL_SAVE_PATH', './src/models/lstm_sentiment_small.pth')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    EPOCHS = 30
    LR = 2e-5
    BATCH_SIZE = 1024

    train_loader = get_dataloader(train_df, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = get_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = get_dataloader(val_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hid_dim=64,
        num_layers=4,
        num_classes=1,
        pad_idx=tokenizer.pad_token_id
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # for param in model.bert.parameters():
    #     param.requires_grad = False

    for epoch in range(EPOCHS):
        train_loss, train_acc, early_break = train(model, train_loader, optimizer, device, num_classes=1, epoch=epoch)
        val_loss, val_acc = evaluate(model, val_loader, device, num_classes=1)
        print(f"[Epoch {epoch+1}/{EPOCHS}]: "
            f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
            f"Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if early_break:
            break
