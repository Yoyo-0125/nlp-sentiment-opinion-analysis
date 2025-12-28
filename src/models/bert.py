import torch.nn as nn
from transformers import BertModel

class BERTSentiment(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits