import random
import json
import torch

from nltk_utils import tokenize, stem
from model_rnn_lstm import RNNLSTMNet   # model baru

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding="utf-8") as json_data:
    intents = json.load(json_data)

# file model hasil training baru
FILE = "data_rnn_lstm.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
word2idx = data["word2idx"]
tags = data["tags"]
num_layers = data["num_layers"]
model_state = data["model_state"]
max_len = data["max_len"]
embedding_dim = data["embedding_dim"]
# bisa juga hardcode sesuai train.py: max_len = panjang sequence terbesar saat training

# load model
model = RNNLSTMNet(
    input_size=input_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=output_size
).to(device)

model.load_state_dict(model_state)
model.eval()

bot_name = "Gleamore"

def encode_sentence(sentence, word2idx, max_len):
    """Ubah kalimat ke urutan indeks + padding"""
    tokens = tokenize(sentence)
    seq = [word2idx.get(stem(w), 0) for w in tokens]  # gunakan 0 jika tidak ada di vocab
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))  # padding
    else:
        seq = seq[:max_len]  # cut jika kepanjangan
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)  # (1, max_len)

def get_response(msg):
    X = encode_sentence(msg, word2idx, max_len)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.90:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Saya tidak paham yang Anda maksud, silahkan bertanya kembali."

if __name__ == "__main__":
    print("Silahkan berkomunikasi! (Ketik 'quit' untuk keluar)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")
