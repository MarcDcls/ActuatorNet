from mlp import MLP
import torch as th
import time
import random
import os

nb_inferences = 1000

# Create fake inputs for windows 2, 5 and 10:
fake_inputs_w2 = []
fake_inputs_w5 = []
fake_inputs_w10 = []

for i in range(nb_inferences):
    inputs = [random.uniform(-1, 1) for _ in range(20)]
    fake_inputs_w2.append(inputs[:4])
    fake_inputs_w5.append(inputs[:10])
    fake_inputs_w10.append(inputs[:20])

# Test inference time for each model:
for model in os.listdir("models"):
    if not model.endswith(".pth"):
        continue

    mlp = MLP.load("models/" + model, device=th.device("cpu"))
    mlp.eval()

    if mlp.dimensions()[0] == 4:
        inputs = fake_inputs_w2
    elif mlp.dimensions()[0] == 10:
        inputs = fake_inputs_w5
    elif mlp.dimensions()[0] == 20:
        inputs = fake_inputs_w10

    start = time.time()
    for input in inputs:
        mlp(input)
    end = time.time()

    print(f"{model}: {(end - start)/nb_inferences*1e6} μs")