"""Графики для отчета."""

import csv
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/20261009-151838")
with (results / "epochs.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))

epochs = list(range(1, 13))
output = Path("assets")
output.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots(figsize=(11, 4))
for run, label in (("one_cycle", "1cycle"), ("inverse_12", "Inverse")):
    values = [
        mean(
            float(row["test_accuracy"])
            for row in rows
            if row["run"] == run and int(row["epoch"]) == epoch
        )
        for epoch in epochs
    ]
    ax.plot(epochs, values, marker="o", markersize=3, label=label)
ax.set(xlim=(1, 12), ylim=(93, 100), xlabel="Эпоха", ylabel="Точность на тесте, %")
ax.set_xticks(epochs)
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(output / "accuracy.png", dpi=160)
plt.close(fig)

lr = [
    float(row["learning_rate"])
    for row in rows
    if row["run"] == "one_cycle" and row["seed"] == "0"
]
assert len(lr) == 12, "Ожидаются 12 эпох для 1cycle, seed 0"
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(epochs, lr, marker="o", markersize=3)
ax.set(xlim=(1, 12), ylim=(0, 0.105), xlabel="Эпоха", ylabel="Learning rate")
ax.set_xticks(epochs)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(output / "learning_rate.png", dpi=160)
plt.close(fig)
