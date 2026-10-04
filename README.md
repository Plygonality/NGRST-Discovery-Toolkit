# NGRST Discovery

A Python research toolkit that turns one Roman-shaped exposure into a sky catalog of known and unknown sources.

Roman calibration stays in `romancal`. This loop starts at a Level 2 rate image or a Level 3 coadd and treats a Level 4 catalog as a comparison table. Science imaging from commissioning is not the input. The committed fixture is Roman-shaped and offline.

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

## Discovery loop

One local command reads a Roman-shaped ASDF, detects sources in the error array, deblends them, attaches RA and Dec from the WCS, and cross-matches an offline reference catalog. Unknowns are ranked first, then by match residual, then by local SNR. `anomaly_score` is a morphology column from an isolation forest. It is not the sort key, and it is not a discovery by itself.

```bash
python -m ngrst discover tests/fixtures/detector_fixture.asdf --top 8
```

Outputs, both gitignored:

```text
data/processed/detector_fixture_catalog.parquet
data/candidates/detector_fixture/001_object_….png
```

A candidate row has RA, Dec, exposure ID, filter, local SNR, reference separation, and a known-versus-unknown mark. Rows without a WCS finish as `pixel_only` and `unclassified`. They are not marked unknown.

The offline reference is `tests/fixtures/reference_catalog.ecsv`. Pass `--reference` to use another local ECSV or parquet file. `--crossmatch-online` is off unless you set it. SIMBAD is not queried during a normal run.

Optional, and off by default. This is the only path that contacts MAST. It asks for one Roman exposure in one filter and one cone, prefers a Level 2 product, downloads that one file into `data/raw/`, and then runs the local loop. An empty result exits without scanning the archive.

```bash
python -m ngrst discover --mast --filter F158 --ra 150.0 --dec 2.0 --radius 0.05
```

What this loop does not do: live survey mining, color outliers, or time-domain measurements.

The older pixel demo is still available. It has no WCS, so every row stays unclassified:

```bash
python src/make_demo.py
python -m ngrst discover data/raw/demo_scene.npz --top 10
```

## Project layout

```text
NGRST-Discovery-Toolkit/
├── src/
│   ├── ngrst.py
│   ├── make_demo.py
│   └── roman_toolkit.py
├── tests/
│   ├── fixtures/
│   │   ├── build_fixture.py
│   │   ├── detector_fixture.asdf
│   │   ├── reference_catalog.ecsv
│   │   └── truth.ecsv
│   ├── test_discovery.py
│   └── test_discovery_sky.py
├── notebooks/
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

The tests use the committed fixture and do not call MAST, Gaia, or SIMBAD.

## License

MIT. See [LICENSE](LICENSE).
