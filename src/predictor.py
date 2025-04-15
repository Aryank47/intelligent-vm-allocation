import torch.nn as nn


# Neural Network Predictor with increased Dropout
class UsagePredictor(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size=1, dropout_prob=0.1
    ):  # noqa
        super(UsagePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        x = self.relu(out)
        out = self.dropout(x)
        out = self.fc2(out)
        return out
