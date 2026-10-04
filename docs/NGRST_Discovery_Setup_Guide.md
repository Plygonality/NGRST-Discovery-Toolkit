# NGRST Discovery Toolkit

## Overview

This guide sets up a Windows-native astronomy analysis toolkit for Roman Space Telescope data.

### Current setup

```text
OS:             Windows 11
Python:         3.12.10
IDE:            PyCharm 2026.2.3
Project folder: E:\NGRST Discovery
Storage:        External SSD
```

You do **not** need VS Code.

You do **not** need Linux or WSL for the current toolkit.

The GitHub repository already contains this toolkit. Clone it and install from `requirements.txt`, or follow the steps below to recreate the project in PyCharm. The listings in this guide match the repository source. Catalog column names follow Photutils 3 (`x_centroid`, `semimajor_axis`, and `n_pixels`).

The current goal is to work with already processed Roman science data, especially L2/L3 products, and build tools for:

- source detection
- photometry
- morphology
- anomaly detection
- candidate ranking
- candidate image export
- later catalog cross-matching
- later multi-filter and time-domain analysis

---

# 1. Open the project in PyCharm

Open PyCharm.

Choose:

```text
Open
```

Navigate to:

```text
E:\NGRST Discovery
```

Open the folder.

If PyCharm asks whether you trust the project, choose:

```text
Trust Project
```

---

# 2. Create a virtual environment

Go to:

```text
File
→ Settings
→ Python Interpreter
```

Click:

```text
Add Interpreter
→ Add Local Interpreter
→ Virtualenv Environment
```

Use:

```text
Environment:      New
Location:         E:\NGRST Discovery\.venv
Base interpreter: Python 3.12.10
```

Leave these disabled:

```text
☐ Inherit global site-packages
☐ Make available to all projects
```

Click:

```text
OK
```

---

# 3. Verify Python

Open the PyCharm terminal.

Run:

```powershell
python --version
```

Expected:

```text
Python 3.12.10
```

Then:

```powershell
where.exe python
```

The active interpreter should point to:

```text
E:\NGRST Discovery\.venv\Scripts\python.exe
```

---

# 4. Upgrade pip

Run:

```powershell
python -m pip install --upgrade pip
```

---

# 5. Install the astronomy stack

Run:

```powershell
python -m pip install numpy scipy pandas matplotlib scikit-learn astropy astroquery photutils asdf pyarrow roman-datamodels
```

Main packages:

| Package | Purpose |
|---|---|
| NumPy | numerical arrays |
| SciPy | scientific algorithms |
| Pandas | catalogs and tables |
| Matplotlib | plots and candidate images |
| scikit-learn | anomaly detection |
| Astropy | astronomy infrastructure |
| Astroquery | MAST and astronomy catalog access |
| Photutils | source detection and photometry |
| ASDF | Roman data format support |
| PyArrow | efficient table storage |
| roman-datamodels | Roman WFI data products |

---

# 6. Verify installation

Run:

```powershell
python -c "import numpy, scipy, pandas, matplotlib, sklearn, astropy, astroquery, photutils, asdf, roman_datamodels; print('NGRST environment OK')"
```

Expected:

```text
NGRST environment OK
```

---

# 7. Create the project structure

Create these folders in PyCharm:

```text
E:\NGRST Discovery
│
├── .venv
├── src
├── data
│   ├── raw
│   ├── processed
│   └── candidates
└── notebooks
```

Purpose:

```text
raw
    input telescope data

processed
    measured catalogs

candidates
    ranked unusual objects

src
    Python source code

notebooks
    exploratory research
```

---

# 8. Create a synthetic test observation

Create:

```text
src\make_demo.py
```

Paste:

```python
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
```

Run it from PyCharm.

Expected:

```text
Synthetic observation created.
Path: E:\NGRST Discovery\data\raw\demo_scene.npz
Resolution: 1024 x 1024
```

---

# 9. Create the discovery engine

Create:

```text
src\roman_toolkit.py
```

Paste:

```python
from pathlib import Path
import argparse

import asdf
import matplotlib
import numpy as np
import pandas as pd
import photutils

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.stats import SigmaClip

from photutils.background import (
    Background2D,
    MedianBackground,
)

from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
)

from sklearn.ensemble import IsolationForest


photutils.future_column_names = True


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIRECTORY = ROOT / "data" / "processed"
CANDIDATE_DIRECTORY = ROOT / "data" / "candidates"


def load_demo_file(path):
    with np.load(path) as file:
        data = np.asarray(
            file["data"],
            dtype=np.float64,
        )

        if "dq" in file:
            dq = np.asarray(
                file["dq"],
                dtype=np.uint32,
            )
        else:
            dq = np.zeros(
                data.shape,
                dtype=np.uint32,
            )

    return data, dq


def load_roman_asdf(path):
    print("Opening Roman ASDF using roman_datamodels...")

    try:
        import roman_datamodels as rdm

        model = rdm.open(path)

        try:
            data = np.asarray(
                model.data,
                dtype=np.float64,
            ).copy()

            dq_data = getattr(
                model,
                "dq",
                None,
            )

            if dq_data is None:
                dq = np.zeros(
                    data.shape,
                    dtype=np.uint32,
                )
            else:
                dq = np.asarray(
                    dq_data,
                    dtype=np.uint32,
                ).copy()

        finally:
            close = getattr(model, "close", None)

            if callable(close):
                close()

        return data, dq

    except Exception as error:
        print("roman_datamodels failed.")
        print(f"Reason: {error}")
        print("Trying generic ASDF reader...")

    with asdf.open(path) as file:
        tree = file.tree

        if "roman" in tree:
            tree = tree["roman"]

        if "data" not in tree:
            raise RuntimeError(
                "No science data array found."
            )

        data = np.asarray(
            tree["data"],
            dtype=np.float64,
        ).copy()

        if "dq" in tree:
            dq = np.asarray(
                tree["dq"],
                dtype=np.uint32,
            ).copy()
        else:
            dq = np.zeros(
                data.shape,
                dtype=np.uint32,
            )

    return data, dq


def load_image(path):
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return load_demo_file(path)

    if suffix == ".asdf":
        return load_roman_asdf(path)

    raise ValueError(
        f"Unsupported file format: {suffix}"
    )


def subtract_background(data, dq):
    mask = (
        (dq != 0)
        |
        ~np.isfinite(data)
    )

    sigma_clip = SigmaClip(
        sigma=3.0
    )

    background = Background2D(
        data,
        box_size=(64, 64),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground(),
        mask=mask,
    )

    clean = data - background.background

    return clean, background, mask


def detect_objects(
    image,
    background,
    mask,
):
    threshold = (
        4.0
        *
        background.background_rms
    )

    segmentation = detect_sources(
        image,
        threshold,
        n_pixels=8,
        mask=mask,
    )

    if segmentation is None:
        raise RuntimeError(
            "No sources detected."
        )

    return segmentation


def build_catalog(
    image,
    segmentation,
    mask,
):
    catalog = SourceCatalog(
        image,
        segmentation,
        mask=mask,
    )

    columns = [
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
    ]

    table = catalog.to_table(
        columns=columns
    )

    return table.to_pandas()


def score_anomalies(df):
    flux = pd.to_numeric(
        df["segment_flux"],
        errors="coerce",
    )

    area = pd.to_numeric(
        df["area"],
        errors="coerce",
    )

    major = pd.to_numeric(
        df["semimajor_axis"],
        errors="coerce",
    )

    minor = pd.to_numeric(
        df["semiminor_axis"],
        errors="coerce",
    )

    eccentricity = pd.to_numeric(
        df["eccentricity"],
        errors="coerce",
    )

    maximum = pd.to_numeric(
        df["max_value"],
        errors="coerce",
    )

    features = pd.DataFrame(
        {
            "log_flux": np.log10(
                np.clip(
                    flux,
                    1e-12,
                    None,
                )
            ),
            "log_area": np.log10(
                np.clip(
                    area,
                    1e-12,
                    None,
                )
            ),
            "eccentricity": eccentricity,
            "axis_ratio":
                minor
                /
                np.clip(
                    major,
                    1e-12,
                    None,
                ),
            "peak_fraction":
                maximum
                /
                np.clip(
                    flux,
                    1e-12,
                    None,
                ),
        }
    )

    valid = np.isfinite(
        features
    ).all(axis=1)

    df["anomaly_score"] = np.nan

    if valid.sum() < 10:
        raise RuntimeError(
            "Too few valid sources for anomaly detection."
        )

    model = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        features.loc[valid]
    )

    scores = (
        -model.decision_function(
            features.loc[valid]
        )
    )

    df.loc[
        valid,
        "anomaly_score"
    ] = scores

    return df.sort_values(
        "anomaly_score",
        ascending=False,
    )


def save_candidates(
    image,
    catalog,
    output_directory,
    number=30,
    half_size=50,
):
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = (
        catalog
        .dropna(
            subset=["anomaly_score"]
        )
        .head(number)
    )

    height, width = image.shape

    for rank, (_, row) in enumerate(
        candidates.iterrows(),
        start=1,
    ):
        x = int(round(row["x_centroid"]))
        y = int(round(row["y_centroid"]))

        x0 = max(0, x - half_size)
        x1 = min(width, x + half_size)

        y0 = max(0, y - half_size)
        y1 = min(height, y + half_size)

        cutout = image[
            y0:y1,
            x0:x1,
        ]

        finite = cutout[
            np.isfinite(cutout)
        ]

        if finite.size == 0:
            continue

        vmin = np.percentile(
            finite,
            5,
        )

        vmax = np.percentile(
            finite,
            99.5,
        )

        plt.figure(
            figsize=(5, 5)
        )

        plt.imshow(
            cutout,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        plt.title(
            f"Candidate #{rank}\n"
            f"Object {int(row['label'])}\n"
            f"Anomaly score "
            f"{row['anomaly_score']:.4f}"
        )

        plt.tight_layout()

        filename = (
            output_directory
            /
            (
                f"{rank:03d}"
                f"_object_"
                f"{int(row['label'])}"
                f".png"
            )
        )

        plt.savefig(
            filename,
            dpi=160,
        )

        plt.close()


def run_pipeline(
    input_path,
    top=30,
):
    print()
    print("NGRST DISCOVERY TOOLKIT")
    print("=======================")
    print()

    print(
        f"Input: {input_path}"
    )

    image, dq = load_image(
        input_path
    )

    print(
        f"Image dimensions: "
        f"{image.shape}"
    )

    print(
        "1. Modelling background..."
    )

    clean, background, mask = (
        subtract_background(
            image,
            dq,
        )
    )

    print(
        "2. Detecting astronomical sources..."
    )

    segmentation = detect_objects(
        clean,
        background,
        mask,
    )

    print(
        "3. Measuring sources..."
    )

    catalog = build_catalog(
        clean,
        segmentation,
        mask,
    )

    print(
        f"   Sources detected: "
        f"{len(catalog):,}"
    )

    print(
        "4. Searching for anomalies..."
    )

    catalog = score_anomalies(
        catalog
    )

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog_path = (
        PROCESSED_DIRECTORY
        /
        (
            input_path.stem
            +
            "_catalog.csv"
        )
    )

    catalog.to_csv(
        catalog_path,
        index=False,
    )

    candidate_path = (
        CANDIDATE_DIRECTORY
        /
        input_path.stem
    )

    print(
        "5. Rendering candidate images..."
    )

    save_candidates(
        clean,
        catalog,
        candidate_path,
        number=top,
    )

    print()
    print("COMPLETE")
    print("========")
    print()

    print(
        f"Catalog:\n"
        f"{catalog_path}"
    )

    print()

    print(
        f"Candidate images:\n"
        f"{candidate_path}"
    )

    print()

    print(
        "Highest anomaly scores:"
    )

    print(
        catalog[
            [
                "label",
                "x_centroid",
                "y_centroid",
                "anomaly_score",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "NGRST astronomical "
            "discovery toolkit"
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Input .npz test data "
            "or Roman .asdf file"
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help=(
            "Number of candidate "
            "objects to export"
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        args.input.resolve(),
        args.top,
    )


if __name__ == "__main__":
    main()
```

---

# 10. Run the discovery engine

In the PyCharm terminal:

```powershell
python src\roman_toolkit.py data\raw\demo_scene.npz
```

Expected structure:

```text
NGRST DISCOVERY TOOLKIT
=======================

Input: E:\NGRST Discovery\data\raw\demo_scene.npz

1. Modelling background...
2. Detecting astronomical sources...
3. Measuring sources...
4. Searching for anomalies...
5. Rendering candidate images...

COMPLETE
```

---

# 11. Inspect candidate objects

Open:

```text
data
└── candidates
    └── demo_scene
```

You should find ranked images:

```text
001_object_....png
002_object_....png
003_object_....png
...
```

The ranking represents statistical unusualness.

A high anomaly score does **not** automatically mean a discovery.

---

# 12. Inspect the catalog

Open:

```text
data\processed\demo_scene_catalog.csv
```

Important columns include:

```text
label
x_centroid
y_centroid
area
segment_flux
max_value
eccentricity
semimajor_axis
semiminor_axis
orientation
anomaly_score
```

---

# 13. Create a PyCharm run configuration

Go to:

```text
Run configuration dropdown
→ Edit Configurations
→ +
→ Python
```

Use:

### Name

```text
NGRST Discovery Demo
```

### Script path

```text
E:\NGRST Discovery\src\roman_toolkit.py
```

### Script parameters

```text
data\raw\demo_scene.npz
```

### Working directory

```text
E:\NGRST Discovery
```

### Python interpreter

```text
E:\NGRST Discovery\.venv\Scripts\python.exe
```

Click:

```text
Apply
→ OK
```

You can now run the pipeline using PyCharm's green Run button.

---

# 14. Add a `.gitignore`

Create:

```text
.gitignore
```

Paste:

```gitignore
.venv/
.idea/
__pycache__/
*.pyc

data/raw/
data/processed/
data/candidates/

.ipynb_checkpoints/
```

Do not commit large telescope datasets to GitHub.

---

# 15. Using real Roman data later

For a calibrated Roman WFI file:

```text
something_cal.asdf
```

put it in:

```text
E:\NGRST Discovery\data\raw\
```

Then run:

```powershell
python src\roman_toolkit.py "data\raw\FILE_NAME.asdf"
```

The toolkit attempts to load the file through:

```text
roman_datamodels
```

and then falls back to generic ASDF access if necessary.

---

# 16. Current architecture

```text
Roman L2/L3 data
        ↓
roman_datamodels / ASDF
        ↓
quality masking
        ↓
background modelling
        ↓
source detection
        ↓
morphological measurements
        ↓
feature vectors
        ↓
Isolation Forest
        ↓
anomaly ranking
        ↓
CSV catalog + candidate PNGs
```

---

# 17. Development roadmap

## Version 0.1

Current build:

```text
✓ Windows environment
✓ synthetic observations
✓ source detection
✓ photometry
✓ morphology
✓ anomaly scoring
✓ candidate cutouts
```

## Version 0.2

Add celestial coordinates:

```text
pixel X/Y
↓
Roman WCS
↓
RA / Dec
```

Then cross-match against:

- SIMBAD
- Gaia
- Pan-STARRS
- 2MASS
- WISE
- MAST

## Version 0.3

Add multi-filter analysis:

```text
F062
F087
F106
F129
F146
F158
F184
F213
```

Calculate color indices and detect spectral outliers.

## Version 0.4

Add time-domain analysis:

```text
observation 1
observation 2
observation 3
        ↓
same sky source
        ↓
compare
```

Measure:

- brightness change
- positional change
- color change
- morphology change

Potential searches:

- supernovae
- variables
- AGN activity
- microlensing
- moving objects
- proper-motion anomalies
- disappearing sources
- transients

## Version 0.5

Add automated MAST access:

```text
MAST
↓
query new Roman observations
↓
download
↓
analyze automatically
↓
rank candidates
↓
store results
```

---

# 18. GitHub repository

Yes, this project is well suited to a GitHub repository.

Recommended repository name:

```text
NGRST-Discovery
```

Alternative:

```text
roman-discovery
```

Recommended short description:

```text
Python toolkit for detecting, ranking, and investigating unusual sources in Nancy Grace Roman Space Telescope data.
```

Recommended public structure:

```text
NGRST-Discovery/
│
├── src/
│   ├── make_demo.py
│   └── roman_toolkit.py
│
├── notebooks/
│
├── tests/
│
├── docs/
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

Do **not** upload:

```text
.venv/
.idea/
data/raw/
data/processed/
data/candidates/
```

Large Roman science products should stay outside Git.

---

# 19. Create `requirements.txt`

From the PyCharm terminal:

```powershell
python -m pip freeze > requirements.txt
```

The repository `requirements.txt` already pins the versions this toolkit was verified with. Install that file instead of replacing it with a fresh freeze, unless you intend to upgrade the stack.

Later, another machine can recreate the environment using:

```powershell
python -m pip install -r requirements.txt
```

---

# 20. Recommended GitHub README opening

```markdown
# NGRST Discovery

A Python research toolkit for detecting, ranking, and investigating unusual astronomical sources in Nancy Grace Roman Space Telescope data.

The project is intended to provide a modular pipeline for:

- Roman ASDF ingestion
- source detection
- photometry
- morphology analysis
- anomaly detection
- candidate ranking
- catalog cross-matching
- multi-filter analysis
- time-domain searches

The current prototype uses synthetic observations to validate the analysis pipeline before Roman science data become available.
```

---

# 21. Recommended next step

Do not add more machine learning yet.

The next technical milestone should be:

```text
Roman ASDF
↓
WCS
↓
RA / Dec
↓
catalog cross-match
↓
known / unknown candidate classification
```

That is the point where the project becomes substantially more useful for real astronomical investigation.
