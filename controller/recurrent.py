"""Existing live GRU/LSTM control model, preserving state_dict names."""
from torch import nn
from vla.nn import mlp


class RNNPolicy(nn.Module):
    state_batch_dim = 1

    def __init__(self, obs_dim, act_dim, cell="gru", hidden=256, layers=1, enc_hidden=128):
        super().__init__()
        self.input_dim = obs_dim
        self.action_dim = act_dim
        self.encoder = mlp([obs_dim, enc_hidden], final_activation=True)
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(enc_hidden, hidden, num_layers=layers, batch_first=True)
        self.head = mlp([hidden, enc_hidden, act_dim])

    def forward(self, x, state=None):
        y, state = self.rnn(self.encoder(x), state)
        return self.head(y), state
