import numpy as np
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

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

X_data = []
y_data = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_data.append(bag)
    label = tags.index(tag)
    y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Shuffle data manual supaya acak
indices = np.arange(len(X_data))
np.random.shuffle(indices)

X_data = X_data[indices]
y_data = y_data[indices]

# Bagi data menjadi train dan validation 80:20
train_size = int(0.8 * len(X_data))

X_train = X_data[:train_size]
y_train = y_data[:train_size]

X_val = X_data[train_size:]
y_val = y_data[train_size:]

# Parameter
num_epochs = 50
batch_size = 8
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

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

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

# Early Stopping setup
patience = 5
best_val_loss = float('inf')
counter = 0

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

    model.eval()
    with torch.no_grad():
        # Hitung val loss dan akurasi
        total_val_loss = 0
        correct = 0
        total = 0
        for words, labels in val_loader:
            words = words.to(device)
            labels = labels.to(device)
            outputs = model(words)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        # Save best model
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping pada epoch ke {epoch+1}")
            break

end_time = time.time()
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)

print(f'Final Loss: {loss.item():.4f}, Final validation loss: {avg_val_loss:.4f}, Final accuracy: {accuracy:.2f}%')
print(f'Training berakhir di {minutes} menit {seconds} detik')

# Simpan model terbaik (best validation loss)
data = {
    "model_state": best_model_state,
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data3.pth"
torch.save(data, FILE)
print(f'Training complete. File saved to {FILE}')
