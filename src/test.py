import torch
import torch.nn.functional as F
from tqdm import tqdm

def test(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    epoch_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc