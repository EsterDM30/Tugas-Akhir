import torch
import torch.nn as nn

class RNNLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding_dim=64):
        super(RNNLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 🧩 Embedding layer untuk ubah index → vektor
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # 🔁 RNN pertama
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'
        )

        # 🔁 LSTM kedua
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 🔚 Fully connected
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) → index kata
        x = self.embedding(x)  # → (batch, seq_len, embedding_dim)

        # Init hidden state untuk RNN
        h0_rnn = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out_rnn, _ = self.rnn(x, h0_rnn)

        # Init hidden state + cell state untuk LSTM
        h0_lstm = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0_lstm = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out_lstm, _ = self.lstm(out_rnn, (h0_lstm, c0_lstm))

        # Ambil time-step terakhir
        out = out_lstm[:, -1, :]
        out = self.fc(out)
        return out
