import torch as th
import wandb
import optparse
from mlp import MLP
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Parse arguments
parser = optparse.OptionParser()
parser.add_option("-s", "--size", dest="window", default=2, type="int", help="window size")
parser.add_option("-n", "--nodes", dest="nodes", default=32, type="int", help="number of nodes per layer")
parser.add_option("-w", "--wandb", dest="wandb", default=0, type="int", help="using wandb")
parser.add_option("-a", "--activation", dest="activation", default="ReLU", type="str", help="activation function")
parser.add_option("-e", "--epochs", dest="epochs", default=300, type="int", help="number of epochs")
args = parser.parse_args()[0]

use_wandb = True if args.wandb == 1 else False
project_name = "actuator-net-w" + str(args.window) + "-n" + str(args.nodes)
model_name = args.activation + "-w" + str(args.window)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

dataset = Dataset.load("data/dataset_w" + str(args.window) + ".npz")
# dataset.shuffle() # No shuffling to have drastically different training and testing datasets
train_dataset, test_dataset = dataset.split(0.8)

training_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True if device == "cuda" else False)
testing_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True if device == "cuda" else False)

if args.activation == "ReLU":
    activation = th.nn.ReLU()
elif args.activation == "Tanh":
    activation = th.nn.Tanh()
elif args.activation == "Softsign":
    activation = th.nn.Softsign()
elif args.activation == "LeakyReLU":
    activation = th.nn.LeakyReLU()
else:
    raise ValueError("Activation function not supported")

mlp = MLP(dataset.inputs_size, dataset.outputs_size, 32, 3, activation, device)
mlp.set_input_scales(dataset.inputs_scales)
mlp.set_output_scales(dataset.outputs_scales)

optimizer = th.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=0)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, verbose=True, threshold=1e-3, threshold_mode="rel", cooldown=0, min_lr=1e-5, eps=1e-8)

loss_func = th.nn.functional.smooth_l1_loss

def train_epoch(net, loader):
    loss_sum = 0
    for batch in tqdm(loader):
        inputs = batch["input"].to(device)
        outputs = batch["output"].to(device)

        loss = loss_func(net(inputs), outputs)
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_sum / len(loader)

def test_epoch(net, loader):
    loss_sum = 0
    for batch in tqdm(loader):
        inputs = batch["input"].to(device)
        outputs = batch["output"].to(device)

        loss = loss_func(net(inputs), outputs)
        loss_sum += loss.item()
    return loss_sum / len(loader)

if use_wandb:
    wandb.init(project=project_name, name=model_name)
    wandb.watch(mlp)

epochs = args.epochs
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs} ...")

    print("Train :")
    mlp.train()
    avg_tloss = train_epoch(mlp, training_loader)
    
    print("Testing :")
    mlp.eval()
    with th.no_grad():
        avg_vloss = test_epoch(mlp, testing_loader)

    if use_wandb:
        wandb.log({"epoch": epoch + 1, "avg_tloss": avg_tloss, "avg_vloss": avg_vloss, "lr": optimizer.param_groups[0]["lr"]})

    scheduler.step(avg_vloss)

    # Saving the model
    # if (epoch + 1) % 5 == 0:
    mlp.save("models/" + model_name + ".pth")