# Data directories

The pipeline creates these folders when it runs. Their contents stay out of Git.

```text
data/
├── raw/          input telescope data and the synthetic demo scene
├── processed/    measured catalogs
└── candidates/   ranked unusual objects
```

Generate the synthetic scene, then run the discovery engine:

```text
python src/make_demo.py
python src/roman_toolkit.py data/raw/demo_scene.npz
```

For a calibrated Roman WFI file, put `something_cal.asdf` in `data/raw/` and pass that path to `roman_toolkit.py`.
