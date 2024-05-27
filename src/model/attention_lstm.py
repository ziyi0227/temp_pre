import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """Attention mechanism module."""

    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # Apply attention mechanism
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return context_vector


class AttentionLSTM(nn.Module):
    """
    LSTM model with attention mechanism.

    Args:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state `h`.
        output_size (int): The number of output features.
        num_layers (int): Number of recurrent layers.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = AttentionModule(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier initialization for better training performance."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        context_vector = self.attention(out)
        out = self.fc(context_vector)
        return out