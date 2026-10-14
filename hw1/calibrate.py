import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, layer_costs, memory


def fit(s, b, measured_time, measured_energy):
    f, q = layer_costs(s, b)
    target_ms = measured_time * 1000

    def residual(x):
        predicted_ms = x[0] + np.maximum(f * x[1] / 1e9, q * x[2] / 1e9).sum(axis=-1)
        return (predicted_ms - target_ms) / target_ms

    candidates = [
        least_squares(residual, start, bounds=([0, 1e-6, 1e-6], np.inf), max_nfev=3000)
        for start in ([0.05, 0.25, 3], [0.1, 1, 10], [0.01, 0.1, 1])
    ]
    best = min(candidates, key=lambda result: result.cost)
    if not best.success:
        raise RuntimeError(f"Latency fit failed: {best.message}")
    theta = {
        "overhead_s": best.x[0] / 1000,
        "flops_per_s": 1e12 / best.x[1],
        "bytes_per_s": 1e12 / best.x[2],
    }
    x = np.column_stack(
        (latency(s, b, theta), flops(s, b) / 1e9, bytes_moved(s, b) / 1e9)
    )
    weighted = x / measured_energy[:, None]
    scale = np.linalg.norm(weighted, axis=0)
    coefficients, _ = nnls(weighted / scale, np.ones(len(s)))
    coefficients /= scale
    theta_energy = {
        "base_power_w": coefficients[0],
        "joules_per_flop": coefficients[1] / 1e9,
        "joules_per_byte": coefficients[2] / 1e9,
        "latency": theta,
    }
    return {
        "latency": theta,
        "energy": theta_energy,
        "energy_design_condition": float(np.linalg.cond(weighted / scale)),
    }


def plot_grid(rows, theta, output):
    sizes = sorted({int(row["S"]) for row in rows})
    batches = np.array(sorted({int(row["B"]) for row in rows}))
    specs = [
        ("flops", "FLOPs (GFLOP)", 1e9, flops),
        ("memory_bytes", "Peak allocated memory (GiB)", 2**30, memory),
        (
            "latency_s",
            "Latency (ms)",
            0.001,
            lambda s, b: latency(s, b, theta["latency"]),
        ),
        ("energy_j", "Energy (J)", 1, lambda s, b: energy(s, b, theta["energy"])),
    ]
    for key, title, scale, predict in specs:
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
        for ax, s in zip(axes.flat, sizes):
            ax.plot(
                batches,
                predict(s, batches) / scale,
                color="black",
                linewidth=1.3,
                label="Equation",
            )
            group = [row for row in rows if int(row["S"]) == s]
            for validation, marker, color, label in [
                (0, "o", "tab:blue", "Calibration"),
                (1, "x", "tab:orange", "Validation"),
            ]:
                selected = [
                    r
                    for r in group
                    if r["status"] == "ok" and int(r["is_validation"]) == validation
                ]
                values = []
                for row in selected:
                    if key == "flops":
                        # The profiler does not count GAP arithmetic or Linear bias.
                        value = float(row["profiler_flops"]) + int(row["B"]) * (
                            2 * s**2 + 356
                        )
                    else:
                        value = float(row[key])
                    values.append(value / scale)
                ax.scatter(
                    [int(r["B"]) for r in selected],
                    values,
                    marker=marker,
                    color=color,
                    s=24,
                    label=label,
                )
            if key == "memory_bytes":
                ordered = sorted(group, key=lambda r: int(r["B"]))
                ax.plot(
                    [int(r["B"]) for r in ordered],
                    [float(r["available_bytes"]) / scale for r in ordered],
                    linestyle=":",
                    color="gray",
                    label="Available budget",
                )
                oom = [r for r in group if r["status"] == "OOM"]
                ax.scatter(
                    [int(r["B"]) for r in oom],
                    [float(r["available_bytes"]) / scale for r in oom],
                    marker="v",
                    color="red",
                    label="OOM (at budget)",
                )
            ax.set(
                title=f"S = {s}",
                xlabel="Batch size B",
                ylabel=title,
                xscale="log",
                yscale="log",
            )
            ax.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(handles, labels, loc="center")
        note = (
            "\nPoints: profiler conv/linear FLOPs + analytic GAP and bias"
            if key == "flops"
            else ""
        )
        fig.suptitle(title + note)
        fig.savefig(output / f"{key}.png", dpi=160)
        plt.close(fig)


def summarize(rows, theta, environment):
    ok = [row for row in rows if row["status"] == "ok"]
    s = np.array([int(r["S"]) for r in ok])
    b = np.array([int(r["B"]) for r in ok])
    validation = np.array([int(r["is_validation"]) for r in ok], dtype=bool)
    predictions = {
        "memory_bytes": memory(s, b),
        "latency_s": latency(s, b, theta["latency"]),
        "energy_j": energy(s, b, theta["energy"]),
    }
    lines = [
        "# Результаты измерений",
        "",
        f"GPU: {environment['gpu']}. PyTorch: {environment['torch']}. CUDA: {environment['cuda']}.",
        "",
    ]
    if environment.get("memory_fraction", 1) < 1:
        lines += [
            f"Дополнительный эксперимент с лимитом аллокатора: {environment['memory_fraction']:.0%} общей памяти GPU, {environment['memory_limit_bytes'] / 2**30:.3f} GiB.",
            "Бюджет на графике учитывает этот лимит. OOM относится к запуску с ограничением памяти.",
            "",
        ]
    lines += [
        f"Успешных конфигураций: {len(ok)} из {len(rows)}. OOM: {len(rows) - len(ok)}.",
        f"В калибровке: {int((~validation).sum())}. В проверке: {int(validation.sum())}.",
        "",
        "| Величина | Выборка | MAPE, % | Медианная ошибка, % | 90-й перцентиль, % |",
        "|---|---|---:|---:|---:|",
    ]
    errors = {}
    for key, predicted in predictions.items():
        errors[key] = 100 * np.abs(
            predicted / np.array([float(r[key]) for r in ok]) - 1
        )
        for label, mask in [("Калибровка", ~validation), ("Проверка", validation)]:
            e = errors[key][mask]
            lines.append(
                f"| {key} | {label} | {e.mean():.2f} | {np.median(e):.2f} | {np.quantile(e, 0.9):.2f} |"
            )
    f, q = layer_costs(s, b)
    compute = f / theta["latency"]["flops_per_s"]
    transfer = q / theta["latency"]["bytes_per_s"]
    components = np.column_stack(
        (
            np.full(len(ok), theta["latency"]["overhead_s"]),
            np.where(compute >= transfer, compute, 0).sum(axis=-1),
            np.where(compute < transfer, transfer, 0).sum(axis=-1),
        )
    )
    regimes = np.array(["launch", "compute", "memory"])[components.argmax(axis=1)]
    lines += [
        "",
        "## Режимы по модели",
        "",
        "Это классификация по наибольшему слагаемому предсказанного времени.",
        "",
        "| Режим | Проверочных точек | MAPE latency, % |",
        "|---|---:|---:|",
    ]
    for regime in ["launch", "memory", "compute"]:
        mask = validation & (regimes == regime)
        error = f"{errors['latency_s'][mask].mean():.2f}" if mask.any() else "нет точек"
        lines.append(f"| {regime} | {mask.sum()} | {error} |")
    lines += [
        "",
        "## Наибольшие ошибки latency на проверке",
        "",
        "| S | B | Измерение, ms | Прогноз, ms | Ошибка, % |",
        "|---:|---:|---:|---:|---:|",
    ]
    worst = np.flatnonzero(validation)[
        np.argsort(errors["latency_s"][validation])[-5:][::-1]
    ]
    for i in worst:
        lines.append(
            f"| {s[i]} | {b[i]} | {float(ok[i]['latency_s']) * 1000:.3f} | {predictions['latency_s'][i] * 1000:.3f} | {errors['latency_s'][i]:.2f} |"
        )
    observed_oom = np.array([r["status"] == "OOM" for r in rows])
    predicted_oom = np.array(
        [memory(int(r["S"]), int(r["B"])) > float(r["available_bytes"]) for r in rows]
    )
    lines += [
        "",
        "## Память и энергия",
        "",
        f"OOM предсказан и наблюдался: {int((predicted_oom & observed_oom).sum())}.",
        f"OOM наблюдался, но не предсказан: {int((~predicted_oom & observed_oom).sum())}.",
        f"OOM предсказан, но запуск прошел: {int((predicted_oom & ~observed_oom).sum())}.",
        "Прогноз OOM использует память тензоров без workspace. Он может недооценивать потребность CUDA. Лимит аллокатора относится к резервируемой памяти; фрагментация и округление могут вызвать OOM раньше.",
        "",
    ]
    if (observed_oom | predicted_oom).any():
        lines += [
            "| S | B | Прогноз памяти, GiB | Бюджет, GiB | Прогноз OOM | Наблюдение |",
            "|---:|---:|---:|---:|---|---|",
        ]
        for row, observed, predicted in zip(rows, observed_oom, predicted_oom):
            if observed or predicted:
                s_i, b_i = int(row["S"]), int(row["B"])
                lines.append(
                    f"| {s_i} | {b_i} | {memory(s_i, b_i) / 2**30:.3f} | {float(row['available_bytes']) / 2**30:.3f} | {'да' if predicted else 'нет'} | {row['status']} |"
                )
        lines.append("")
    cvs = np.array([float(r["energy_cv"]) for r in ok])
    ratios = np.array(
        [float(r["energy_window_latency_s"]) / float(r["latency_s"]) for r in ok]
    )
    lines += [
        f"Медианный CV энергии по трем окнам: {100 * np.median(cvs):.2f}%. Максимум: {100 * cvs.max():.2f}%.",
        f"Медианное отношение времени прохода в энергетическом окне к latency: {np.median(ratios):.3f}.",
        f"Число обусловленности матрицы энергетической модели: {theta['energy_design_condition']:.3g}.",
        "",
        "Коэффициенты являются эффективными параметрами модели. При плохой обусловленности нельзя интерпретировать их как независимые физические характеристики GPU.",
        "",
        "FLOPs профилировщика проверены для каждой успешной точки. В график добавлен аналитический вклад GAP и bias, который профилировщик не считает.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    args = parser.parse_args()
    with args.results.joinpath("measurements.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    environment = json.loads(args.results.joinpath("environment.json").read_text())
    points = {(int(r["S"]), int(r["B"])) for r in rows}
    if (
        len(rows) != 132
        or len(points) != 132
        or points != {tuple(p) for p in environment["grid"]}
    ):
        parser.error("Expected all 132 configurations from one complete run")
    for row in rows:
        if row["status"] not in {"ok", "OOM"} or row["is_validation"] not in {"0", "1"}:
            parser.error("Invalid status or validation flag in measurements.csv")
        if row["status"] == "ok":
            for key in ["latency_s", "memory_bytes", "energy_j"]:
                if not np.isfinite(float(row[key])) or float(row[key]) <= 0:
                    parser.error(f"Invalid {key} at S={row['S']}, B={row['B']}")
    train = [r for r in rows if r["status"] == "ok" and r["is_validation"] == "0"]
    validation = [r for r in rows if r["status"] == "ok" and r["is_validation"] == "1"]
    if len(train) < 4 or not validation:
        parser.error("Not enough successful calibration or validation points")
    theta = fit(
        np.array([int(r["S"]) for r in train]),
        np.array([int(r["B"]) for r in train]),
        np.array([float(r["latency_s"]) for r in train]),
        np.array([float(r["energy_j"]) for r in train]),
    )
    args.results.joinpath("theta.json").write_text(json.dumps(theta, indent=2) + "\n")
    figures = args.results / "figures"
    figures.mkdir(exist_ok=True)
    plot_grid(rows, theta, figures)
    print(summarize(rows, theta, environment))


if __name__ == "__main__":
    main()
