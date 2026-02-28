import torch
import torch.nn as nn

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # hanya ambil output dari time-step terakhir
        return out
