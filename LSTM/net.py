import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.relu = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.fc3(x)
