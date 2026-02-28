import torch
import torch.nn as nn

class CNNNet(nn.Module):
    def __init__(self, num_classes, height, width):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Hitung otomatis size fitur output conv layer
        self._to_linear = None
        self._get_conv_output((1, height, width))

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            bs = 1
            input = torch.rand(bs, *shape)
            output_feat = self.pool(self.relu(self.conv1(input)))
            n_size = output_feat.view(bs, -1).shape[1]
            self._to_linear = n_size
        return n_size

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = out.view(out.size(0), -1)  # flatten
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out