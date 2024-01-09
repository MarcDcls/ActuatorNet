import torch as th

class MLP(th.nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, hidden_dimension: int = 32, hidden_layers: int = 3):
        super().__init__()

        layers = [th.nn.Linear(input_dimension, hidden_dimension)]
        for i in range(hidden_layers):
            layers.append(th.nn.ReLU())
            layers.append(th.nn.Linear(hidden_dimension, hidden_dimension))
        layers.append(th.nn.Linear(hidden_dimension, output_dimension))

        self.net = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
