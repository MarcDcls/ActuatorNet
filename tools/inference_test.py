from mlp import MLP
import torch as th
import time
import random
import os
from dataset import Dataset

nb_inferences = 3000

# Create fake inputs for windows 2, 5 and 10:
fake_inputs_w2 = []
fake_inputs_w5 = []
fake_inputs_w10 = []
fake_inputs_w20 = []

for i in range(nb_inferences):
    inputs = [random.uniform(-1, 1) for _ in range(40)]
    fake_inputs_w2.append(inputs[:4])
    fake_inputs_w5.append(inputs[:10])
    fake_inputs_w10.append(inputs[:20])
    fake_inputs_w20.append(inputs)

# Test inference time for each model:
results = []
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
    elif mlp.dimensions()[0] == 40:
        inputs = fake_inputs_w20

    # window = mlp.dimensions()[0]/2
    # dataset = Dataset.load("data/dataset_w" + str(window) + ".npz")
    # train_dataset, test_dataset = dataset.split(0.8) # Not seen in training data

    # for i in range(min(nb_inferences, len(test_dataset))):
    #     input = test_dataset[i]
        
    start = time.time()
    for input in inputs:
        mlp(input)
    end = time.time()

    if len(results) == 0:
        results.append((model, (end - start)/nb_inferences*1e6))
    else:
        for i in range(len(results)):
            if (end - start)/nb_inferences*1e6 < results[i][1]:
                results.insert(i, (model, (end - start)/nb_inferences*1e6))
                break
        else:
            results.append((model, (end - start)/nb_inferences*1e6))

for model, time in results:
    print(f"{model}: {time:.2f} µs")