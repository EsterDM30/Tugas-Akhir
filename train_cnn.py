import numpy as np
import random
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from nltk_utils import bag_of_words, tokenize, stem
from model_cnn import CNNNet  # CNN model

with open('intents.json', 'r') as f:
    intents = json.load(f)

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

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X = []
y = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split ke train dan validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape untuk CNN: (batch_size, 1, height, width)
input_vector_length = X.shape[1]
height = 12
width = input_vector_length // height
if height * width != input_vector_length:
    raise ValueError(f"Input size {input_vector_length} tidak cocok untuk reshape ke ({height},{width})")

X_train = X_train.reshape(-1, 1, height, width)
X_val = X_val.reshape(-1, 1, height, width)

# Parameter
num_epochs = 50
batch_size = 8
learning_rate = 0.01
output_size = len(tags)

print(f'Input shape: {X_train.shape}')
print(f'Output size: {output_size}')

# Dataset class
class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)
        self.n_samples = len(x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNNet(num_classes=output_size, height=height, width=width).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Early Stopping Setup
patience = 5
best_val_loss = float('inf')
counter = 0

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss_total = 0
        correct = 0
        total = 0
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

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Check early stopping
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

print(f'Final Train loss: {loss.item():.4f}, Final Val Loss: {avg_val_loss:.4f},Final accuracy: {val_accuracy:.2f}%')
print(f'Training berakhir di {minutes} menit {seconds} detik')

# Save model
data = {
    "model_state": model.state_dict(),
    "input_shape": (1, height, width),
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data_cnn2.pth"
torch.save(data, FILE)
print(f'Training complete. File saved to {FILE}')
