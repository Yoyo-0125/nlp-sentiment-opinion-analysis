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

from dataset import read_dataset, get_dataloader
from models.bert import BERTClassifier
from models.lstm import LSTMClassifier


def train(model, train_loader, optimizer, device, num_classes, epoch):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

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

    epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, device, num_classes=1):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
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

    epoch_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    data_path = os.getenv('DATASET_PATH', './data')
    df = read_dataset(os.path.join(data_path, 'weibo_senti_100k.csv'))
    # df = df.sample(n=10000, random_state=42)
    print('Dataset size:', len(df))
    print(df['text'].str.len().describe())

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    test_df, val_df = train_test_split(temp_df, test_size=0.2, random_state=42, stratify=temp_df['label'])

    # 从环境变量读取 tokenizer 路径
    tokenizer_path = os.getenv('ROBERTA_MODEL_PATH', './src/models/chinese-roberta-wwm-ext')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    EPOCHS = 30
    LR = 2e-5
    BATCH_SIZE = 512

    train_loader = get_dataloader(train_df, tokenizer, batch_size=BATCH_SIZE, shuffle=True, max_len=256, num_workers=12)
    test_loader = get_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False, max_len=256, num_workers=12)
    val_loader = get_dataloader(val_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False, max_len=256, num_workers=12)

    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        hid_dim=128,
        num_layers=4,
        num_classes=1,
        pad_idx=tokenizer.pad_token_id,
        dropout=0.3
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # for param in model.bert.parameters():
    #     param.requires_grad = False

    train_losses, train_acces = [], []
    val_losses, val_acces = [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, device, num_classes=1, epoch=epoch)
        val_loss, val_acc = evaluate(model, val_loader, device, num_classes=1)
        
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        val_losses.append(val_loss)
        val_acces.append(val_acc)

        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'lstm_classifier.pth')
            print('Model saved.')

        print(f"[Epoch {epoch+1}/{EPOCHS}]: "
            f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
            f"Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")