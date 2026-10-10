"""Reproduce the MNIST/LeNet experiment from Smith and Topin (2018)."""

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 512
RUNS = (("one_cycle", 12), ("inverse_12", 12), ("inverse_85", 85))


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)
        for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = nn.functional.max_pool2d(self.conv1(x), 2)
        x = nn.functional.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(x, 1)
        return self.fc2(nn.functional.relu(self.fc1(x)))


def make_optimizer(model):
    weights = [p for name, p in model.named_parameters() if not name.endswith("bias")]
    biases = [p for name, p in model.named_parameters() if name.endswith("bias")]
    # Caffe's LeNet uses twice the learning rate for biases.
    return SGD(
        [{"params": weights, "lr": 0.01}, {"params": biases, "lr": 0.02}],
        momentum=0.9,
        weight_decay=0.0005,
    )


def make_scheduler(optimizer, run_name, steps_per_epoch, epochs):
    if run_name == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=[0.1, 0.2],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=5 / 12,
            anneal_strategy="linear",
            base_momentum=0.8,
            max_momentum=0.95,
            div_factor=10,
            final_div_factor=100,
            three_phase=True,
        )
    return LambdaLR(optimizer, lambda step: (1 + 0.0001 * step) ** -0.75)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return 100 * correct / len(loader.dataset)


def train_run(name, epochs, seed, train_data, test_loader, device, writer):
    torch.manual_seed(seed)
    loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = LeNet().to(device)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, name, len(loader), epochs)
    loss_fn = nn.CrossEntropyLoss()
    elapsed = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        sync(device)
        start = time.perf_counter()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            last_lr = optimizer.param_groups[0]["lr"]
            last_momentum = optimizer.param_groups[0]["momentum"]
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach() * len(labels)
        sync(device)
        elapsed += time.perf_counter() - start
        last_accuracy = test_accuracy(model, test_loader, device)
        writer.writerow(
            {
                "run": name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": total_loss.item() / len(train_data),
                "test_accuracy": last_accuracy,
                "train_seconds": elapsed,
                "learning_rate": last_lr,
                "momentum": last_momentum,
            }
        )
        print(
            f"{name} seed={seed} epoch={epoch}/{epochs} "
            f"test={last_accuracy:.2f}% train={elapsed:.1f}s",
            flush=True,
        )
    return {
        "run": name,
        "seed": seed,
        "epochs": epochs,
        "test_accuracy": last_accuracy,
        "train_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path("results") / stamp
    output.mkdir(parents=True)

    # Caffe's MNIST input scale is 1/256, while ToTensor alone uses 1/255.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * (255 / 256)),
        ]
    )
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    (output / "environment.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "batch_size": BATCH_SIZE,
                "seeds": args.seeds,
            },
            indent=2,
        )
        + "\n"
    )

    with (
        (output / "epochs.csv").open("w", newline="", buffering=1) as file,
        (output / "summary.csv").open("w", newline="", buffering=1) as summary_file,
    ):
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run",
                "seed",
                "epoch",
                "train_loss",
                "test_accuracy",
                "train_seconds",
                "learning_rate",
                "momentum",
            ],
        )
        writer.writeheader()
        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "run",
                "seed",
                "epochs",
                "test_accuracy",
                "train_seconds",
            ],
        )
        summary_writer.writeheader()
        for seed in args.seeds:
            for name, epochs in RUNS:
                summary_writer.writerow(
                    train_run(
                        name, epochs, seed, train_data, test_loader, device, writer
                    )
                )
    print(f"Results saved in {output}")


if __name__ == "__main__":
    main()
