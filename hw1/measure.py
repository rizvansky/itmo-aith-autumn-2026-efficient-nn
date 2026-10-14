import argparse
import csv
import gc
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

from equations import PARAMETERS, flops, memory
from models import make_model

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 42


def grid():
    rng = np.random.default_rng(SEED)
    extra_s = rng.choice(
        [s for s in range(32, 513, 16) if s not in BASE_S], 4, replace=False
    )
    extra_b = rng.choice(
        [b for b in range(1, 257) if b not in BASE_B], 3, replace=False
    )
    sizes = sorted(BASE_S + extra_s.tolist())
    batches = sorted(BASE_B + extra_b.tolist())
    points = list(itertools.product(sizes, batches))
    rng.shuffle(points)
    return points


@torch.inference_mode()
def energy_window(model, x, handle, seconds, method):
    torch.cuda.synchronize()
    start = time.perf_counter()
    if method == "counter":
        initial = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    else:
        times = [0.0]
        powers = [pynvml.nvmlDeviceGetPowerUsage(handle) / 1000]
    count = 0
    elapsed = 0.0
    while elapsed < seconds:
        model(x)
        torch.cuda.synchronize()
        count += 1
        elapsed = time.perf_counter() - start
        if method == "power" and elapsed - times[-1] >= 0.05:
            powers.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
            times.append(time.perf_counter() - start)
    if method == "counter":
        joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - initial) / 1000
    else:
        powers.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
        times.append(time.perf_counter() - start)
        joules = float(
            np.sum(np.diff(times) * (np.array(powers[:-1]) + powers[1:]) / 2)
        )
    duration = time.perf_counter() - start
    if joules <= 0 or not np.isfinite(joules):
        raise RuntimeError(
            "NVML returned invalid energy. Try --energy-method power in a new output folder."
        )
    return {"joules": joules, "seconds": duration, "forwards": count}


@torch.inference_mode()
def measure_point(model, s, b, handle, args, method):
    x = torch.randn(b, 3, s, s, device="cuda", dtype=torch.float32)
    for _ in range(5):
        model(x)
    torch.cuda.synchronize()
    # Warm latency and energy measurements under the same sustained load.
    start = time.perf_counter()
    while time.perf_counter() - start < 2:
        model(x)
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()

    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    windows = [
        energy_window(model, x, handle, args.energy_seconds, method) for _ in range(3)
    ]
    energies = np.array([w["joules"] / w["forwards"] for w in windows])

    # Profiling is outside the memory, time and energy measurements.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as prof:
        model(x)
        torch.cuda.synchronize()
    profiled_flops = sum(event.flops for event in prof.key_averages())
    expected_mac_flops = b * (17712 * s**2 + 313344)
    if profiled_flops != expected_mac_flops:
        raise RuntimeError(f"Profiler FLOPs {profiled_flops} != {expected_mac_flops}")

    return {
        "status": "ok",
        "latency_s": float(np.median(timings)),
        "latency_q25_s": float(np.quantile(timings, 0.25)),
        "latency_q75_s": float(np.quantile(timings, 0.75)),
        "memory_bytes": peak,
        "reserved_memory_bytes": reserved_peak,
        "baseline_bytes": baseline,
        "profiler_flops": profiled_flops,
        "energy_j": float(np.median(energies)),
        "energy_cv": float(np.std(energies, ddof=1) / np.mean(energies)),
        "energy_window_latency_s": float(
            np.median([w["seconds"] / w["forwards"] for w in windows])
        ),
        "energy_method": method,
        "latency_samples_s": json.dumps(timings),
        "energy_windows": json.dumps(windows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--energy-seconds", type=float, default=3.0)
    parser.add_argument("--energy-method", choices=["auto", "power"], default="auto")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=1.0,
        help="Allocator limit as a fraction of total GPU memory; use 0.20 for extra OOM runs",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Only two small configurations"
    )
    args = parser.parse_args()
    if (
        args.repeats < 3
        or not np.isfinite(args.energy_seconds)
        or args.energy_seconds < 2
    ):
        parser.error("Use at least 3 timing samples and 2 seconds per energy window")
    if not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1]")
    if args.output.joinpath("measurements.csv").exists():
        parser.error("measurements.csv already exists; use a new --output folder")
    if not torch.cuda.is_available():
        parser.error(
            "A CUDA GPU is required. In Colab select Runtime > Change runtime type > T4 GPU."
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    pynvml.nvmlInit()
    try:
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
        memory_limit = int(properties.total_memory * args.memory_fraction)
        gpu_uuid = f"GPU-{properties.uuid}"
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        method = "power"
        if args.energy_method == "auto":
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                method = "counter"
            except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_FunctionNotFound):
                pass
        if method == "power":
            pynvml.nvmlDeviceGetPowerUsage(handle)
        model = make_model().cuda().eval()
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS
        points = [(32, 1), (64, 2)] if args.smoke else grid()
        source_dir = Path(__file__).resolve().parent
        environment = {
            "gpu": properties.name,
            "gpu_uuid": gpu_uuid,
            "gpu_total_bytes": properties.total_memory,
            "memory_fraction": args.memory_fraction,
            "memory_limit_bytes": memory_limit,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "seed": SEED,
            "dtype": "float32",
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": False,
            "matmul_allow_tf32": False,
            "repeats": args.repeats,
            "warmup_seconds": 2.0,
            "energy_seconds": args.energy_seconds,
            "energy_repeats": 3,
            "energy_method": method,
            "smoke": args.smoke,
            "grid": points,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_sha256": {
                name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                for name in ["models.py", "equations.py", "measure.py"]
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        args.output.joinpath("environment.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
        fields = [
            "S",
            "B",
            "is_validation",
            "status",
            "latency_s",
            "latency_q25_s",
            "latency_q75_s",
            "memory_bytes",
            "reserved_memory_bytes",
            "baseline_bytes",
            "predicted_memory_bytes",
            "flops",
            "profiler_flops",
            "energy_j",
            "energy_cv",
            "energy_window_latency_s",
            "energy_method",
            "available_bytes",
            "latency_samples_s",
            "energy_windows",
        ]
        with args.output.joinpath("measurements.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, (s, b) in enumerate(points, 1):
                gc.collect()
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                row = {
                    "S": s,
                    "B": b,
                    "is_validation": int(s not in BASE_S or b not in BASE_B),
                    "predicted_memory_bytes": int(memory(s, b)),
                    "flops": int(flops(s, b)),
                    # The budget is limited by both free memory and the allocator quota.
                    "available_bytes": min(
                        free + torch.cuda.memory_allocated(), memory_limit
                    ),
                }
                try:
                    row.update(measure_point(model, s, b, handle, args, method))
                except torch.cuda.OutOfMemoryError:
                    row["status"] = "OOM"
                writer.writerow(row)
                stream.flush()
                detail = (
                    f"{row['latency_s'] * 1000:.3f} ms, {row['energy_j']:.5f} J"
                    if row["status"] == "ok"
                    else "OOM"
                )
                print(f"[{i:3}/{len(points)}] S={s:3}, B={b:3}: {detail}", flush=True)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
