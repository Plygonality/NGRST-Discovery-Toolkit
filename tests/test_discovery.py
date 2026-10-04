import subprocess
import sys
from pathlib import Path

import asdf
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import roman_toolkit


PLANTED = (
    (250, 300),
    (730, 680),
    (620, 220),
    (400, 800),
)


def test_load_image_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file format"):
        roman_toolkit.load_image(tmp_path / "scene.fits")


def test_demo_pipeline_ranks_planted_outliers(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    candidates = tmp_path / "candidates"
    monkeypatch.setattr(roman_toolkit, "PROCESSED_DIRECTORY", processed)
    monkeypatch.setattr(roman_toolkit, "CANDIDATE_DIRECTORY", candidates)

    completed = subprocess.run(
        [sys.executable, str(SRC / "make_demo.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = ROOT / "data" / "raw" / "demo_scene.npz"
    assert generated.is_file()
    assert "Synthetic observation created." in completed.stdout
    assert "1024 x 1024" in completed.stdout

    with np.load(generated) as scene_file:
        assert scene_file["data"].shape == (1024, 1024)
        assert scene_file["dq"].shape == (1024, 1024)
        assert scene_file["dq"].dtype == np.uint32

    roman_toolkit.run_pipeline(generated, top=10)

    catalog = pd.read_csv(processed / "demo_scene_catalog.csv")
    assert list(catalog.columns) == [
        "label",
        "x_centroid",
        "y_centroid",
        "area",
        "segment_flux",
        "max_value",
        "eccentricity",
        "semimajor_axis",
        "semiminor_axis",
        "orientation",
        "anomaly_score",
    ]
    assert len(catalog) > 100
    scored = catalog["anomaly_score"].dropna()
    assert scored.is_monotonic_decreasing
    assert len(scored) == len(catalog)

    images = sorted((candidates / "demo_scene").glob("*.png"))
    assert len(images) == 10
    assert images[0].name.startswith("001_object_")

    top = catalog.head(15)
    found = 0
    for x, y in PLANTED:
        nearby = (
            (top["x_centroid"] - x).abs().lt(20)
            & (top["y_centroid"] - y).abs().lt(20)
        )
        found += int(nearby.any())
    assert found >= 2


def test_generic_asdf_fallback(tmp_path):
    path = tmp_path / "scene.asdf"
    image = np.full((32, 32), 10.0)
    dq = np.zeros((32, 32), dtype=np.uint32)
    dq[0, 0] = 1

    with asdf.AsdfFile({"roman": {"data": image, "dq": dq}}) as handle:
        handle.write_to(path)

    data, loaded_dq = roman_toolkit.load_image(path)
    assert data.shape == (32, 32)
    assert data[1, 1] == 10.0
    assert loaded_dq[0, 0] == 1
