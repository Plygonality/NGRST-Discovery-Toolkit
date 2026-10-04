"""Build the offline Roman-shaped detector fixture.

The file is an ImageModel ASDF: data, dq, err, a TAN gwcs, F158, and
exposure FIXTURE-001. Science pixels are synthetic. Nothing is downloaded.
"""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.modeling import models
from astropy.table import Table
import gwcs
from gwcs import coordinate_frames as cf
from roman_datamodels.datamodels import ImageModel


HERE = Path(__file__).resolve().parent
HEIGHT = 256
WIDTH = 256
CENTER = 127.5
SCALE_DEG = 0.11 / 3600.0

UNKNOWNS = (
    (48.0, 210.0, "UNK-1"),
    (70.0, 70.0, "UNK-2"),
    (188.0, 48.0, "UNK-3"),
    (210.0, 188.0, "UNK-4"),
)
KNOWNS = (
    (60.0, 150.0, "REF-1"),
    (110.0, 48.0, "REF-2"),
    (150.0, 170.0, "REF-3"),
    (200.0, 110.0, "REF-4"),
)
DQ_CENTER = (30.0, 30.0)
DQ_HALF = 12


def make_gwcs():
    shift = models.Shift(-CENTER) & models.Shift(-CENTER)
    scale = models.Scale(-SCALE_DEG) & models.Scale(SCALE_DEG)
    transform = (
        shift
        | scale
        | models.Pix2Sky_TAN()
        | models.RotateNative2Celestial(150.0, 2.0, 180.0)
    )
    detector = cf.Frame2D(
        name="detector",
        axes_names=("x", "y"),
        unit=(u.pix, u.pix),
    )
    sky = cf.CelestialFrame(
        reference_frame=None,
        name="icrs",
        unit=(u.deg, u.deg),
        axes_names=("ra", "dec"),
    )
    return gwcs.WCS(
        forward_transform=transform,
        input_frame=detector,
        output_frame=sky,
    )


def add_gaussian(image, x, y, amplitude, sigma):
    radius = int(max(sigma * 6, 3))
    x0 = max(0, int(x) - radius)
    x1 = min(WIDTH, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(HEIGHT, int(y) + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    image[y0:y1, x0:x1] += amplitude * np.exp(
        -(((xx - x) ** 2) + ((yy - y) ** 2)) / (2 * sigma**2)
    )


def ordinary_positions(reserved):
    rng = np.random.default_rng(7)
    chosen = []
    while len(chosen) < 40:
        x = float(rng.uniform(20, WIDTH - 20))
        y = float(rng.uniform(20, HEIGHT - 20))
        if all((x - px) ** 2 + (y - py) ** 2 >= 18**2 for px, py in reserved + chosen):
            chosen.append((x, y))
    return chosen


def world(wcs, x, y):
    sky = wcs.pixel_to_world(x, y)
    return float(sky.ra.deg), float(sky.dec.deg)


def build():
    wcs = make_gwcs()
    data = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    reserved = [(x, y) for x, y, _ in UNKNOWNS + KNOWNS]
    reserved.append(DQ_CENTER)
    ordinary = ordinary_positions(reserved)

    for x, y in ordinary:
        add_gaussian(data, x, y, amplitude=8.2, sigma=1.45)
    for x, y, _ in KNOWNS:
        add_gaussian(data, x, y, amplitude=16.0, sigma=1.45)
    for x, y, _ in UNKNOWNS:
        add_gaussian(data, x, y, amplitude=55.0, sigma=1.7)
    add_gaussian(data, DQ_CENTER[0], DQ_CENTER[1], amplitude=400.0, sigma=1.3)

    dq = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    x_slice = slice(int(DQ_CENTER[0] - DQ_HALF), int(DQ_CENTER[0] + DQ_HALF))
    y_slice = slice(int(DQ_CENTER[1] - DQ_HALF), int(DQ_CENTER[1] + DQ_HALF))
    dq[y_slice, x_slice] = 1

    err = np.ones((HEIGHT, WIDTH), dtype=np.float32)

    model = ImageModel.create_fake_data(shape=(HEIGHT, WIDTH))
    model.data = data.astype(model.data.dtype, copy=False)
    model.err = err.astype(model.err.dtype, copy=False)
    model.dq = dq.astype(model.dq.dtype, copy=False)
    model.meta.instrument.optical_element = "F158"
    model.meta.observation.observation_id = "FIXTURE-001"
    model.meta.wcs = wcs

    asdf_path = HERE / "detector_fixture.asdf"
    model.save(asdf_path)

    truth_rows = []
    reference_rows = []
    for kind, rows in (("unknown", UNKNOWNS), ("known", KNOWNS)):
        for x, y, source_id in rows:
            ra, dec = world(wcs, x, y)
            truth_rows.append(
                {
                    "source_id": source_id,
                    "kind": kind,
                    "x": x,
                    "y": y,
                    "ra": ra,
                    "dec": dec,
                }
            )
            if kind == "known":
                reference_rows.append(
                    {"source_id": source_id, "ra": ra, "dec": dec}
                )
    ra, dec = world(wcs, DQ_CENTER[0], DQ_CENTER[1])
    truth_rows.append(
        {
            "source_id": "DQ-1",
            "kind": "dq",
            "x": DQ_CENTER[0],
            "y": DQ_CENTER[1],
            "ra": ra,
            "dec": dec,
        }
    )

    Table(rows=truth_rows).write(
        HERE / "truth.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    Table(rows=reference_rows).write(
        HERE / "reference_catalog.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    print(f"Wrote {asdf_path}")
    print(f"Wrote {HERE / 'truth.ecsv'}")
    print(f"Wrote {HERE / 'reference_catalog.ecsv'}")


if __name__ == "__main__":
    build()
