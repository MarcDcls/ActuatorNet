import torch as th
import numpy as np
import os

class MLP(th.nn.Module):
    """
    Multi-layer perceptron.

    Args:
        input_dimension (int): input dimension
        output_dimension (int): output dimension
        device (th.device): device
        hidden_dimension (int): hidden dimension
        hidden_layers (int): number of hidden layers
        activation (th.nn.Module): activation function
        residual (int): number of residual connections
        deterministic (bool): deterministic or probabilistic output
    """
    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 hidden_dimension: int = 32, 
                 hidden_layers: int = 3, 
                 activation: th.nn.Module = th.nn.ReLU(),
                 device: th.device = th.device("cpu")
                 ):
        super().__init__()

        self.device: th.device = device

        layers = [th.nn.Linear(input_dimension, hidden_dimension)]
        layers.append(activation)
        for _ in range(hidden_layers):
            layers.append(th.nn.Linear(hidden_dimension, hidden_dimension))
            layers.append(activation)
        layers.append(th.nn.Linear(hidden_dimension, output_dimension))
        self.net = th.nn.Sequential(*layers).to(device)

        self.input_scales = th.nn.Parameter(th.ones(input_dimension, device=device).to(device), requires_grad=False)
        self.output_scales = th.nn.Parameter(th.ones(output_dimension, device=device).to(device), requires_grad=False)

    def forward(self, x):
        if type(x) is not th.Tensor:
            x = th.Tensor(x).to(self.device)

        y = self.net(x / self.input_scales)

        return y * self.output_scales
    
    def set_input_scales(self, scales: list) -> None:
        """
        Sets the input scales.

        Args:
            scales (list): input scales
        """
        self.input_scales = th.nn.Parameter(th.Tensor(scales).to(self.device), requires_grad=False)

    def set_output_scales(self, scales: list) -> None:
        """
        Sets the output scales.

        Args:
            scales (list): output scales
        """
        self.output_scales = th.nn.Parameter(th.Tensor(scales).to(self.device), requires_grad=False)

    def dimensions(self) -> (int, int):
        """
        Returns the input and output dimensions.
        """
        return self.input_scales.size(0), self.output_scales.size(0)
    
    def save(self, filename: str) -> None:
        """
        Save the model.

        Args:   
            filename (str): file name
        """
        th.save(self, filename)

    @classmethod
    def load(cls, filename: str, device: th.device = th.device("cpu")) -> "MLP":
        """
        Load the model.

        Args:
            filename (str): file name
        """
        mlp = th.load(filename, map_location=device)
        mlp.device = device
        return mlp
        
if __name__ == '__main__':
    mlp = MLP(2, 2, activation=th.nn.Softsign())
    mlp.set_input_scales([2, 3])
    mlp.set_output_scales([4, 5])

    mlp.save("models/test.pth")

    loaded_mlp = MLP.load("models/test.pth", device=th.device("cpu"))
    print(loaded_mlp.input_scales)
    print(loaded_mlp.output_scales)
    print(loaded_mlp.output_scales.device)
