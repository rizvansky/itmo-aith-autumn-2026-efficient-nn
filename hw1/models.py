from torch import nn


def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(32, 64, 5, padding=2, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 100),
    )


if __name__ == "__main__":
    import torch

    from equations import PARAMETERS

    model = make_model().eval()
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    with torch.inference_mode():
        for s, b in [(32, 1), (64, 2)]:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
            ) as prof:
                output = model(torch.randn(b, 3, s, s))
            assert output.shape == (b, 100)
            assert sum(event.flops for event in prof.key_averages()) == b * (
                17_712 * s**2 + 313_344
            )
    print("Model: parameters, output shapes and profiler FLOPs OK")
