import numpy as np
import random
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from nltk_utils import tokenize, stem
from model_rnn_lstm import RNNLSTMNet

# ====== Load intents ======
with open('intents.json', 'r', encoding="utf-8") as f:
    intents = json.load(f)

# ====== Preprocessing ======
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words")

# ====== Buat word2idx ======
word2idx = {w: i for i, w in enumerate(all_words)}

# ====== Encode patterns ======
def encode_sentence(tokens, word2idx):
    return [word2idx[stem(w)] for w in tokens if stem(w) in word2idx]

X = []
y = []

for (pattern_sentence, tag) in xy:
    encoded = encode_sentence(pattern_sentence, word2idx)
    if len(encoded) == 0:
        continue
    X.append(encoded)
    y.append(tags.index(tag))

# Padding sequence (agar sama panjang)
max_len = max(len(seq) for seq in X)
X_padded = np.zeros((len(X), max_len), dtype=np.int64)
for i, seq in enumerate(X):
    X_padded[i, :len(seq)] = seq

y = np.array(y)

# Split train-validation
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

# ====== Hyperparameters ======
num_epochs = 100
batch_size = 8
learning_rate = 0.001
input_size = len(all_words)
hidden_size = 128
output_size = len(tags)
num_layers = 1

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# ====== Dataset class ======
class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.long)
        self.y_data = torch.tensor(y_data, dtype=torch.long)
        self.n_samples = len(x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# ====== Device & Model ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNLSTMNet(input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ====== Training loop ======
patience = 10
best_val_loss = float('inf')
counter = 0

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Evaluasi validation loss & akurasi
    model.eval()
    val_loss_total = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for words, labels in val_loader:
            words = words.to(device)
            labels = labels.to(device)
            outputs = model(words)
            val_loss = criterion(outputs, labels)
            val_loss_total += val_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss_total / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping pada epoch ke {epoch+1}")
            break

end_time = time.time()
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)

print(f'Final Train Loss: {avg_train_loss:.4f}, Final Val Loss: {avg_val_loss:.4f}, '
      f'Final Val Accuracy: {val_accuracy:.2f}%')
print(f'Training berakhir di {minutes} menit {seconds} detik')

# ====== Save model ======
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "word2idx": word2idx,
    "tags": tags,
    "num_layers": num_layers,
    "max_len": max_len,
    "embedding_dim": 64
}

FILE = "data_rnn_lstm.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
