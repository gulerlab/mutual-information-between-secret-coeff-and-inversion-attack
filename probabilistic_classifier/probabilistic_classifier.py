import torch
import torch.nn as nn


class ProbabilisticClassifier(nn.Module):
    def __init__(self, num_input_features, hidden_size_arr, num_output_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_layer = nn.Linear(num_input_features, hidden_size_arr[0])
        init_act = nn.ReLU()
        self.init_layer = nn.Sequential(init_layer, init_act)

        hidden_layers = []
        for idx in range(1, len(hidden_size_arr)):
            hidden_layers.append(nn.Linear(hidden_size_arr[idx - 1], hidden_size_arr[idx]))
            hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.out_layer = nn.Linear(hidden_size_arr[-1], num_output_features)

    def forward(self, input_data):
        init_out = self.init_layer(input_data)
        hidden_out = self.hidden_layers(init_out)
        return self.out_layer(hidden_out)


class BinaryProbabilisticClassifierWithSigmoidAndClamp(nn.Module):
    def __init__(self, num_input_features, hidden_size_arr, num_output_features, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_layer = nn.Linear(num_input_features, hidden_size_arr[0])
        init_act = nn.ReLU()
        self.init_layer = nn.Sequential(init_layer, init_act)

        hidden_layers = []
        for idx in range(1, len(hidden_size_arr)):
            hidden_layers.append(nn.Linear(hidden_size_arr[idx - 1], hidden_size_arr[idx]))
            hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.out_layer = nn.Linear(hidden_size_arr[-1], num_output_features)
        self.sigmoid = nn.Sigmoid()
        self.tau = tau

    def forward(self, input_data):
        init_out = self.init_layer(input_data)
        hidden_out = self.hidden_layers(init_out)
        network_out = self.out_layer(hidden_out)
        probabilities = self.sigmoid(network_out)
        return torch.clamp(probabilities, min=self.tau, max=(1-self.tau))
