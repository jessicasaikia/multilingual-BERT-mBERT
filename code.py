!pip install transformers torch pandas scikit-learn

import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data = pd.read_csv("/content/mBERT.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

data.head()


tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

unique_labels = data["POS_Tags"].str.split().explode().unique()
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
num_labels = len(label_to_id)



def tokenize_and_align_labels(sentence, pos_tags):
    tokenized_inputs = tokenizer(sentence, padding="max_length", truncation=True, return_tensors="pt")
    labels = []

    for word, pos_tag in zip(sentence.split(), pos_tags.split()):
        word_tokens = tokenizer.tokenize(word)
        labels.extend([pos_tag] + ["O"] * (len(word_tokens) - 1))

    labels = labels[:tokenizer.model_max_length]
    labels += ["O"] * (tokenizer.model_max_length - len(labels))

    label_ids = [label_to_id.get(label, -100) for label in labels]
    return tokenized_inputs, label_ids

train_tokenized = [(tokenize_and_align_labels(row['Sentence'], row['POS_Tags'])) for _, row in train_data.iterrows()]
test_tokenized = [(tokenize_and_align_labels(row['Sentence'], row['POS_Tags'])) for _, row in test_data.iterrows()]


class CodeMixedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, labels = self.data[idx]
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(labels)
        }

train_dataset = CodeMixedDataset(train_tokenized)
test_dataset = CodeMixedDataset(test_tokenized)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)


model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(2):
    print(f"Epoch {epoch + 1}")
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss}")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(predictions)
        all_labels.extend(labels)

for i, (pred, label) in enumerate(zip(all_preds[:10], all_labels[:10])):
    predicted_tags = [id_to_label[id.item()] for id in pred if id.item() != -100] 
    true_tags = [id_to_label[id.item()] for id in label if id.item() != -100]
    print(f"Sentence {i + 1}:")
    print(f"Predicted: {predicted_tags}")
    print(f"Actual:    {true_tags}\n")


model.save_pretrained("custom_pos_model")
tokenizer.save_pretrained("custom_pos_model")
