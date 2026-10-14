import numpy as np

PARAMETERS = 1_040_324


def inputs(image_size, batch):
    s, b = np.broadcast_arrays(
        np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float)
    )
    if not np.all(np.isfinite(s) & (s >= 32) & (s <= 512) & (s % 16 == 0)):
        raise ValueError("image_size must be a multiple of 16 in [32, 512]")
    if not np.all(np.isfinite(b) & (b >= 1) & (b <= 256) & (b % 1 == 0)):
        raise ValueError("batch must be an integer in [1, 256]")
    return s, b


def flops(image_size, batch):
    s, b = inputs(image_size, batch)
    return b * (17_714 * s**2 + 313_700)


def memory(image_size, batch):
    s, b = inputs(image_size, batch)
    # Memory of tensors; cuDNN workspaces are not included.
    return 4 * PARAMETERS + 68 * b * s**2


def bytes_moved(image_size, batch):
    s, b = inputs(image_size, batch)
    return 4 * PARAMETERS + b * (380 * s**2 + 8_592)


def layer_costs(image_size, batch):
    """FLOPs and ideal traffic; last axis follows the 17 non-view operations."""
    s, b = inputs(image_size, batch)
    bs2 = (b * s**2)[..., None]
    b = b[..., None]
    f = bs2 * np.array(
        [2352, 0, 0, 6400, 0, 2304, 0, 1024, 0, 4608, 0, 1024, 0, 2, 0, 0, 0]
    )
    f += b * np.array([0] * 14 + [262400, 0, 51300])
    q = bs2 * np.array([44, 64, 56, 24, 32, 24, 16, 24, 32, 20, 8, 12, 16, 8, 0, 0, 0])
    q += b * np.array([0] * 13 + [2048, 3072, 2048, 1424])
    q += 4 * np.array(
        [
            4704,
            0,
            0,
            51200,
            0,
            73728,
            0,
            32768,
            0,
            589824,
            0,
            131072,
            0,
            0,
            131328,
            0,
            25700,
        ]
    )
    return f, q


def latency(image_size, batch, theta):
    f, q = layer_costs(image_size, batch)
    return theta["overhead_s"] + np.maximum(
        f / theta["flops_per_s"], q / theta["bytes_per_s"]
    ).sum(axis=-1)


def energy(image_size, batch, theta_energy):
    return (
        theta_energy["base_power_w"]
        * latency(image_size, batch, theta_energy["latency"])
        + theta_energy["joules_per_flop"] * flops(image_size, batch)
        + theta_energy["joules_per_byte"] * bytes_moved(image_size, batch)
    )


if __name__ == "__main__":
    s = np.arange(32, 513, 16)[:, None]
    b = np.arange(1, 257)[None, :]
    f, q = layer_costs(s, b)
    assert np.array_equal(f.sum(axis=-1), flops(s, b))
    assert np.array_equal(q.sum(axis=-1), bytes_moved(s, b))
    assert flops(32, 1) == 18_452_836
    assert memory(32, 1) == 4_230_928
    assert bytes_moved(32, 1) == 4_559_008
    print("Equations: scalar values and broadcasting OK")
