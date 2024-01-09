import numpy as np
import torch as th
import os
from mlp import MLP
from data import Sample, Dataset, process_logs

window_size = 5

process_logs("logs")
dataset = Dataset(window_size=window_size)
for sample_name in os.listdir("data"):
    sample = Sample()
    sample.load("data/" + sample_name)
    dataset.add(sample)

net = MLP(dataset.input_size, dataset.output_size, 32, 3)
batch_size = 512
optimizer = th.optim.Adam(net.parameters(), 1e-3)

training_ratio = 0.8
iterations = int(np.ceil(len(dataset) / batch_size))
training_iterations = int(np.ceil(iterations * training_ratio))

for i in range(training_iterations):
    print(f"Training {i + 1}/{training_iterations} ...")
    for j in range(batch_size):
        index = np.random.randint(0, len(dataset))
        input, output = dataset[index]
        input = th.tensor(input, dtype=th.float)
        output = th.tensor(output, dtype=th.float)
        loss = th.nn.functional.smooth_l1_loss(net(input), output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

        
for k in range(1024):
    angles = np.random.uniform([-np.pi, 0.2], [np.pi, 1.7], size=(batch_size, 2))
    laser_pos = [model.laser(*angle) for angle in angles]

    angles = th.tensor(angles, dtype=th.float)
    laser_pos = th.tensor(laser_pos, dtype=th.float)

    loss = th.nn.functional.smooth_l1_loss(net(laser_pos), angles)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    
th.save(net.state_dict(), "actuator_model.pth")