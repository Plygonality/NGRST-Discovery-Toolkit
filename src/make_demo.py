from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "demo_scene.npz"

HEIGHT = 1024
WIDTH = 1024

rng = np.random.default_rng(42)


image = rng.normal(
    loc=1000.0,
    scale=5.0,
    size=(HEIGHT, WIDTH),
)

dq = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.uint32,
)


def add_gaussian(
    x,
    y,
    amplitude,
    sigma_x,
    sigma_y,
):
    radius = int(max(sigma_x, sigma_y) * 6)

    x0 = max(0, int(x) - radius)
    x1 = min(WIDTH, int(x) + radius + 1)

    y0 = max(0, int(y) - radius)
    y1 = min(HEIGHT, int(y) + radius + 1)

    yy, xx = np.mgrid[y0:y1, x0:x1]

    source = amplitude * np.exp(
        -(
            ((xx - x) ** 2) / (2 * sigma_x**2)
            +
            ((yy - y) ** 2) / (2 * sigma_y**2)
        )
    )

    image[y0:y1, x0:x1] += source


for _ in range(750):
    x = rng.uniform(30, WIDTH - 30)
    y = rng.uniform(30, HEIGHT - 30)

    amplitude = rng.lognormal(
        mean=5.0,
        sigma=0.7,
    )

    sigma = rng.uniform(
        1.2,
        2.8,
    )

    add_gaussian(
        x,
        y,
        amplitude,
        sigma,
        sigma,
    )


# Deliberately unusual objects

add_gaussian(
    250,
    300,
    900,
    10,
    2,
)

add_gaussian(
    730,
    680,
    300,
    15,
    13,
)

add_gaussian(
    620,
    220,
    5000,
    1,
    1,
)

add_gaussian(
    400,
    800,
    250,
    18,
    2.5,
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    OUTPUT,
    data=image,
    dq=dq,
)

print("Synthetic observation created.")
print(f"Path: {OUTPUT}")
print(f"Resolution: {WIDTH} x {HEIGHT}")
