# NGRST Discovery

A Python research toolkit for detecting, ranking, and investigating unusual astronomical sources in Nancy Grace Roman Space Telescope data.

The project is a modular pipeline for:

- Roman ASDF ingestion
- source detection
- photometry
- morphology analysis
- anomaly detection
- candidate ranking
- catalog cross-matching
- multi-filter analysis
- time-domain searches

The current prototype uses synthetic observations to validate the analysis pipeline before Roman science data are used.

## Install

Python 3.12 is the intended interpreter. Create a virtual environment in the project folder, then install the astronomy stack.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the stack:

```bash
python -c "import numpy, scipy, pandas, matplotlib, sklearn, astropy, astroquery, photutils, asdf, roman_datamodels; print('NGRST environment OK')"
```

PyCharm setup for a Windows workstation is in [docs/NGRST_Discovery_Setup_Guide.md](docs/NGRST_Discovery_Setup_Guide.md).

## Quick start

Create a 1024×1024 synthetic scene with ordinary stars and a few deliberately unusual objects:

```bash
python src/make_demo.py
```

Run the discovery engine:

```bash
python src/roman_toolkit.py data/raw/demo_scene.npz
```

Outputs:

```text
data/processed/demo_scene_catalog.csv
data/candidates/demo_scene/001_object_….png
```

The catalog is ranked by `anomaly_score`. A high score means the source is statistically unusual in this image. It does not by itself mean a discovery.

Export fewer cutouts with `--top`:

```bash
python src/roman_toolkit.py data/raw/demo_scene.npz --top 10
```

## Real Roman data

Place a calibrated Roman WFI file in `data/raw/`, then run:

```bash
python src/roman_toolkit.py "data/raw/FILE_NAME.asdf"
```

The loader tries `roman_datamodels` first and falls back to a generic ASDF read when that open fails. Large Roman science products stay outside Git.

## Pipeline

```text
Roman L2/L3 data or synthetic .npz
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

Catalog columns:

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

Anomaly features are log flux, log area, eccentricity, axis ratio, and peak fraction.

## Project layout

```text
NGRST-Discovery-Toolkit/
├── src/
│   ├── make_demo.py
│   └── roman_toolkit.py
├── notebooks/
├── tests/
├── docs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── candidates/
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

`data/raw/`, `data/processed/`, and `data/candidates/` are gitignored.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

## Roadmap

Version 0.1, this build:

- synthetic observations
- source detection
- photometry
- morphology
- anomaly scoring
- candidate cutouts

Version 0.2: Roman WCS, RA/Dec, and cross-matches to SIMBAD, Gaia, Pan-STARRS, 2MASS, WISE, and MAST.

Version 0.3: multi-filter colors (F062 through F213) and spectral outliers.

Version 0.4: time-domain changes in brightness, position, color, and morphology.

Version 0.5: automated MAST queries, download, analysis, and stored rankings.

The next milestone is coordinates and known-versus-unknown classification, not more machine learning.

## License

MIT. See [LICENSE](LICENSE).
