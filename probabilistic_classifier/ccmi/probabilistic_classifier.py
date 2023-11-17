import torch
import torch.nn as nn


class ProbabilisticClassifier(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.fc_01 = nn.Linear(in_dim, h_dim)
        self.act_01 = nn.ReLU()
        self.fc_02 = nn.Linear(h_dim, h_dim)
        self.act_02 = nn.ReLU()
        self.fc_03 = nn.Linear(h_dim, out_dim)

    def forward(self, input_data):
        hidden_out = self.fc_01(input_data)
        hidden_out = self.act_01(hidden_out)
        hidden_out = self.fc_02(hidden_out)
        hidden_out = self.act_02(hidden_out)
        out = self.fc_03(hidden_out)
        return out
