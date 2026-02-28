import random
import json
import torch

from model_cnn import CNNNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data_cnn2.pth"
data = torch.load(FILE)

input_shape = data["input_shape"]  # (1, height, width)
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = CNNNet(num_classes=output_size, height=input_shape[1], width=input_shape[2]).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, 1, input_shape[1], input_shape[2])  # CNN input shape
    X = torch.from_numpy(X).float().to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.90:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Saya tidak paham..."

if __name__ == "__main__":
    print("Silahkan berkomunikasi! (Ketik 'quit' untuk keluar)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        resp = get_response(sentence)
        print(resp)
