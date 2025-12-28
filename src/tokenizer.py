from transformers import BertTokenizer



encoding = tokenizer(
    "我今天很开心",
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print(encoding)