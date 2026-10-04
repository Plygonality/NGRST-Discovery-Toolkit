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
