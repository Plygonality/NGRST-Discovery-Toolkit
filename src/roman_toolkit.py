from pathlib import Path
import argparse
from dataclasses import dataclass

import asdf
import matplotlib
import numpy as np
import pandas as pd
import photutils

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u

from photutils.background import (
    Background2D,
    MedianBackground,
)
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
)
from sklearn.ensemble import IsolationForest


photutils.future_column_names = True


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIRECTORY = ROOT / "data" / "processed"
CANDIDATE_DIRECTORY = ROOT / "data" / "candidates"
RAW_DIRECTORY = ROOT / "data" / "raw"
DEFAULT_REFERENCE = ROOT / "tests" / "fixtures" / "reference_catalog.ecsv"

CATALOG_COLUMNS = [
    "label",
    "x_centroid",
    "y_centroid",
    "ra",
    "dec",
    "area",
    "segment_flux",
    "max_value",
    "eccentricity",
    "semimajor_axis",
    "semiminor_axis",
    "orientation",
    "local_snr",
    "peak_fraction",
    "anomaly_score",
    "coord_status",
    "match_status",
    "separation_arcsec",
    "filter",
    "exposure_id",
]

EXTRA_COLUMNS = [
    "detect_status",
    "ref_id",
]

MEASUREMENT_COLUMNS = [
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


@dataclass
class Frame:
    data: np.ndarray
    dq: np.ndarray
    err: np.ndarray | None = None
    wcs: WCS | None = None
    filter: str | None = None
    exposure_id: str | None = None
    unit: str | None = None
    source: str = "npz"


def empty_catalog():
    frame = pd.DataFrame(columns=CATALOG_COLUMNS + EXTRA_COLUMNS)
    return frame


def _text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return text


def _meta_get(meta, *keys):
    node = meta
    for key in keys:
        if node is None:
            return None
        if isinstance(node, dict):
            node = node.get(key)
        else:
            node = getattr(node, key, None)
    return _text(node)


def _wcs_from_header_text(text):
    text = _text(text)
    if text is None:
        return None
    header = Header.fromstring(text, sep="\n")
    wcs = WCS(header)
    if getattr(wcs, "has_celestial", False):
        return wcs.celestial
    return None


def _astropy_wcs_from_gwcs(gwcs_obj, shape):
    if not hasattr(gwcs_obj, "pixel_to_world"):
        return None
    ny, nx = shape[-2], shape[-1]
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    try:
        center = gwcs_obj.pixel_to_world(x0, y0)
        right = gwcs_obj.pixel_to_world(x0 + 1.0, y0)
        up = gwcs_obj.pixel_to_world(x0, y0 + 1.0)
    except Exception:
        return None
    if not hasattr(center, "ra"):
        return None
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = [x0 + 1.0, y0 + 1.0]
    wcs.wcs.crval = [float(center.ra.deg), float(center.dec.deg)]
    wcs.wcs.cd = [
        [
            float(right.ra.deg - center.ra.deg),
            float(up.ra.deg - center.ra.deg),
        ],
        [
            float(right.dec.deg - center.dec.deg),
            float(up.dec.deg - center.dec.deg),
        ],
    ]
    return wcs


def coerce_wcs(wcs_obj, shape):
    if wcs_obj is None:
        return None
    if isinstance(wcs_obj, WCS):
        if getattr(wcs_obj, "has_celestial", False):
            return wcs_obj.celestial
        return None
    return _astropy_wcs_from_gwcs(wcs_obj, shape)


def _finalize(catalog):
    for column in CATALOG_COLUMNS + EXTRA_COLUMNS:
        if column not in catalog.columns:
            catalog[column] = np.nan
    ordered = CATALOG_COLUMNS + [
        column for column in EXTRA_COLUMNS if column in catalog.columns
    ]
    return catalog.loc[:, ordered].reset_index(drop=True)


def load_demo_file(path):
    with np.load(path) as file:
        data = np.asarray(file["data"], dtype=np.float64)
        if "dq" in file:
            dq = np.asarray(file["dq"], dtype=np.uint32)
        else:
            dq = np.zeros(data.shape, dtype=np.uint32)
        err = None
        if "err" in file:
            err = np.asarray(file["err"], dtype=np.float64)
        wcs = None
        if "wcs" in file:
            wcs = _wcs_from_header_text(file["wcs"])
        filter_name = _text(file["filter"]) if "filter" in file else None
        exposure_id = (
            _text(file["exposure_id"]) if "exposure_id" in file else None
        )
        unit = _text(file["unit"]) if "unit" in file else None
    return Frame(
        data=data,
        dq=dq,
        err=err,
        wcs=wcs,
        filter=filter_name,
        exposure_id=exposure_id,
        unit=unit,
        source="npz",
    )


def _frame_from_roman_model(model):
    data = np.asarray(model.data, dtype=np.float64).copy()
    dq_data = getattr(model, "dq", None)
    if dq_data is None:
        dq = np.zeros(data.shape, dtype=np.uint32)
    else:
        dq = np.asarray(dq_data, dtype=np.uint32).copy()
    err_data = getattr(model, "err", None)
    err = None if err_data is None else np.asarray(err_data, dtype=np.float64).copy()
    meta = getattr(model, "meta", None)
    wcs_obj = getattr(meta, "wcs", None) if meta is not None else None
    unit = None
    data_unit = getattr(model.data, "unit", None)
    if data_unit is not None:
        unit = str(data_unit)
    return Frame(
        data=data,
        dq=dq,
        err=err,
        wcs=coerce_wcs(wcs_obj, data.shape),
        filter=_meta_get(meta, "instrument", "optical_element"),
        exposure_id=_meta_get(meta, "observation", "observation_id"),
        unit=unit,
        source="asdf",
    )


def _load_generic_asdf(path):
    with asdf.open(path) as file:
        tree = file.tree
        roman = tree["roman"] if "roman" in tree else tree
        if "data" not in roman:
            raise RuntimeError("No science data array found.")
        data = np.asarray(roman["data"], dtype=np.float64).copy()
        if "dq" in roman and roman["dq"] is not None:
            dq = np.asarray(roman["dq"], dtype=np.uint32).copy()
        else:
            dq = np.zeros(data.shape, dtype=np.uint32)
        err = None
        if "err" in roman and roman["err"] is not None:
            err = np.asarray(roman["err"], dtype=np.float64).copy()
        meta = roman.get("meta") if hasattr(roman, "get") else None
        wcs = None
        if "wcs_header" in roman:
            wcs = _wcs_from_header_text(roman["wcs_header"])
        elif "wcs" in roman:
            wcs = coerce_wcs(roman["wcs"], data.shape)
        unit = _text(roman.get("unit")) if hasattr(roman, "get") else None
    return Frame(
        data=data,
        dq=dq,
        err=err,
        wcs=wcs,
        filter=_meta_get(meta, "instrument", "optical_element"),
        exposure_id=_meta_get(meta, "observation", "observation_id"),
        unit=unit,
        source="asdf",
    )


def load_roman_asdf(path):
    print("Opening Roman ASDF using roman_datamodels...")
    try:
        import roman_datamodels as rdm

        model = rdm.open(path)
        try:
            return _frame_from_roman_model(model)
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()
    except Exception as error:
        print("roman_datamodels failed.")
        print(f"Reason: {error}")
        print("Trying generic ASDF reader...")
    return _load_generic_asdf(path)


def load_fits(path):
    with fits.open(path) as hdul:
        image_hdu = None
        for hdu in hdul:
            if hdu.data is None:
                continue
            if np.ndim(hdu.data) >= 2:
                image_hdu = hdu
                break
        if image_hdu is None:
            raise RuntimeError("No image HDU found.")
        data = np.asarray(image_hdu.data, dtype=np.float64)
        if data.ndim > 2:
            data = np.asarray(data[0], dtype=np.float64)
        header = image_hdu.header
        wcs = WCS(header)
        if data.ndim == 2 and wcs.naxis > 2:
            wcs = wcs.celestial
        if not getattr(wcs, "has_celestial", False):
            wcs = None
        err = None
        dq = np.zeros(data.shape, dtype=np.uint32)
        for hdu in hdul:
            name = (hdu.name or "").strip().upper()
            if hdu.data is None or name == "":
                continue
            if name in {"ERR", "ERROR"}:
                err = np.asarray(hdu.data, dtype=np.float64)
            elif name == "DQ":
                dq = np.asarray(hdu.data, dtype=np.uint32)
        filter_name = _text(header.get("FILTER")) or _text(
            header.get("OPTICAL_ELEMENT")
        )
        exposure_id = (
            _text(header.get("OBSERVATION_ID"))
            or _text(header.get("OBS_ID"))
            or _text(header.get("EXPOSURE"))
        )
        unit = _text(header.get("BUNIT"))
    return Frame(
        data=data,
        dq=dq,
        err=err,
        wcs=wcs,
        filter=filter_name,
        exposure_id=exposure_id,
        unit=unit,
        source="fits",
    )


def load_image(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_demo_file(path)
    if suffix == ".asdf":
        return load_roman_asdf(path)
    if suffix == ".fits":
        return load_fits(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def subtract_background(data, dq):
    mask = (dq != 0) | ~np.isfinite(data)
    sigma_clip = SigmaClip(sigma=3.0)
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


def detect_objects(image, background, mask):
    threshold = 4.0 * background.background_rms
    segmentation = detect_sources(
        image,
        threshold,
        n_pixels=8,
        mask=mask,
    )
    if segmentation is None:
        raise RuntimeError("No sources detected.")
    return segmentation


def build_catalog(image, segmentation, mask):
    catalog = SourceCatalog(
        image,
        segmentation,
        mask=mask,
    )
    table = catalog.to_table(columns=MEASUREMENT_COLUMNS)
    return table.to_pandas()


def _local_snr(segmentation, err, labels, flux):
    segments = np.asarray(segmentation.data)
    scores = []
    for label, value in zip(labels, flux):
        pixels = err[segments == int(label)]
        if pixels.size == 0 or not np.isfinite(value):
            scores.append(np.nan)
            continue
        denom = np.sqrt(np.sum(np.square(pixels)))
        if not np.isfinite(denom) or denom <= 0:
            scores.append(np.nan)
        else:
            scores.append(float(value) / float(denom))
    return scores


def detect_sources_sigma(data, err, dq, nsigma=4, npixels=8):
    data = np.asarray(data, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.uint32)
    mask = (
        (dq != 0)
        | ~np.isfinite(data)
        | ~np.isfinite(err)
        | (err <= 0)
    )
    threshold = nsigma * err
    segmentation = detect_sources(
        data,
        threshold,
        n_pixels=npixels,
        mask=mask,
    )
    if segmentation is None:
        return empty_catalog()
    segmentation = deblend_sources(
        data,
        segmentation,
        n_pixels=npixels,
        progress_bar=False,
    )
    catalog = build_catalog(data, segmentation, mask)
    flux = pd.to_numeric(catalog["segment_flux"], errors="coerce")
    peak = pd.to_numeric(catalog["max_value"], errors="coerce")
    catalog["peak_fraction"] = peak / flux.clip(lower=1e-12)
    catalog["local_snr"] = _local_snr(
        segmentation,
        err,
        catalog["label"].to_numpy(),
        flux.to_numpy(),
    )
    catalog["detect_status"] = "error_array"
    return catalog


def assign_sky(catalog, frame):
    catalog = catalog.copy()
    if len(catalog) == 0:
        return _finalize(catalog)
    if frame.wcs is None:
        catalog["ra"] = np.nan
        catalog["dec"] = np.nan
        catalog["coord_status"] = "pixel_only"
        return catalog
    ra, dec = frame.wcs.pixel_to_world_values(
        catalog["x_centroid"].to_numpy(dtype=float),
        catalog["y_centroid"].to_numpy(dtype=float),
    )
    catalog["ra"] = np.asarray(ra, dtype=float)
    catalog["dec"] = np.asarray(dec, dtype=float)
    finite = np.isfinite(catalog["ra"]) & np.isfinite(catalog["dec"])
    catalog["coord_status"] = np.where(finite, "sky", "pixel_only")
    return catalog


def score_anomalies(df):
    df = df.copy()
    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = np.nan
    else:
        df["anomaly_score"] = np.nan
    if len(df) < 10 or "segment_flux" not in df.columns:
        return df

    flux = pd.to_numeric(df["segment_flux"], errors="coerce")
    area = pd.to_numeric(df["area"], errors="coerce")
    major = pd.to_numeric(df["semimajor_axis"], errors="coerce")
    minor = pd.to_numeric(df["semiminor_axis"], errors="coerce")
    eccentricity = pd.to_numeric(df["eccentricity"], errors="coerce")
    maximum = pd.to_numeric(df["max_value"], errors="coerce")
    features = pd.DataFrame(
        {
            "log_flux": np.log10(np.clip(flux, 1e-12, None)),
            "log_area": np.log10(np.clip(area, 1e-12, None)),
            "eccentricity": eccentricity,
            "axis_ratio": minor / np.clip(major, 1e-12, None),
            "peak_fraction": maximum / np.clip(flux, 1e-12, None),
        }
    )
    valid = np.asarray(np.isfinite(features).all(axis=1))
    valid_index = features.index[valid]
    if len(valid_index) < 10:
        print(
            "Too few valid sources for anomaly detection. "
            "anomaly_score left empty."
        )
        return df

    model = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features.loc[valid_index])
    scores = -model.decision_function(features.loc[valid_index])
    df.loc[valid_index, "anomaly_score"] = scores
    return df


def read_reference(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = Table.read(path).to_pandas()
    rename = {column: column.lower() for column in table.columns}
    return table.rename(columns=rename)


def crossmatch(catalog, reference, radius_arcsec=0.3):
    catalog = catalog.copy()
    catalog["match_status"] = "unclassified"
    catalog["separation_arcsec"] = np.nan
    catalog["ref_id"] = pd.Series([None] * len(catalog), dtype="object")
    if len(catalog) == 0:
        return catalog

    on_sky = (
        catalog["coord_status"].astype(str).eq("sky")
        & np.isfinite(pd.to_numeric(catalog["ra"], errors="coerce"))
        & np.isfinite(pd.to_numeric(catalog["dec"], errors="coerce"))
    )
    if reference is None or not bool(on_sky.any()):
        catalog.loc[on_sky, "match_status"] = "unknown"
        return catalog

    reference = reference.copy()
    reference.columns = [column.lower() for column in reference.columns]
    detected = catalog.loc[on_sky]
    catalog_coords = SkyCoord(
        ra=detected["ra"].to_numpy(dtype=float) * u.deg,
        dec=detected["dec"].to_numpy(dtype=float) * u.deg,
    )
    reference_coords = SkyCoord(
        ra=reference["ra"].to_numpy(dtype=float) * u.deg,
        dec=reference["dec"].to_numpy(dtype=float) * u.deg,
    )
    indexes, separations, _ = catalog_coords.match_to_catalog_sky(
        reference_coords
    )
    separation_arcsec = separations.arcsec
    if "source_id" in reference.columns:
        ref_ids = reference["source_id"].astype(str).to_numpy()
    else:
        ref_ids = reference.index.astype(str).to_numpy()

    for position, index, separation in zip(
        detected.index,
        indexes,
        separation_arcsec,
    ):
        if separation <= radius_arcsec:
            catalog.loc[position, "match_status"] = "known"
            catalog.loc[position, "separation_arcsec"] = float(separation)
            catalog.loc[position, "ref_id"] = ref_ids[int(index)]
        else:
            catalog.loc[position, "match_status"] = "unknown"
            catalog.loc[position, "separation_arcsec"] = np.nan
            catalog.loc[position, "ref_id"] = None
    return catalog


def query_simbad_region(ra, dec, radius_arcsec):
    from astroquery.simbad import Simbad

    simbad = Simbad()
    return simbad.query_region(
        SkyCoord(ra, dec, unit="deg"),
        radius=radius_arcsec * u.arcsec,
    )


def crossmatch_online(catalog, radius_arcsec=0.3):
    print("Online cross-match requested.")
    catalog = catalog.copy()
    try:
        on_sky = catalog["coord_status"].astype(str).eq("sky")
        if not bool(on_sky.any()):
            print("Online cross-match skipped: no sky coordinates.")
            return catalog
        for index in catalog.index[on_sky]:
            result = query_simbad_region(
                float(catalog.at[index, "ra"]),
                float(catalog.at[index, "dec"]),
                radius_arcsec,
            )
            if result is None or len(result) == 0:
                continue
            catalog.at[index, "match_status"] = "known"
            catalog.at[index, "ref_id"] = "simbad"
        return catalog
    except Exception as error:
        print(f"Online cross-match failed: {error}")
        print("Local match status kept.")
        return catalog


def rank_catalog(catalog):
    if len(catalog) == 0:
        return catalog
    status_order = {"unknown": 0, "known": 1, "unclassified": 2}
    ranked = catalog.copy()
    ranked["_status"] = (
        ranked["match_status"].map(status_order).fillna(9)
    )
    ranked["_sep"] = pd.to_numeric(
        ranked["separation_arcsec"],
        errors="coerce",
    ).fillna(-np.inf)
    ranked["_snr"] = pd.to_numeric(
        ranked["local_snr"],
        errors="coerce",
    ).fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_status", "_sep", "_snr"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return ranked.drop(columns=["_status", "_sep", "_snr"]).reset_index(drop=True)


def _measure(frame, nsigma):
    if frame.err is None:
        clean, background, mask = subtract_background(frame.data, frame.dq)
        rms = np.asarray(background.background_rms, dtype=np.float64)
        catalog = detect_sources_sigma(
            clean,
            rms,
            frame.dq,
            nsigma=nsigma,
        )
        catalog["local_snr"] = np.nan
        catalog["detect_status"] = "rms_fallback"
        return clean, catalog
    catalog = detect_sources_sigma(
        frame.data,
        frame.err,
        frame.dq,
        nsigma=nsigma,
    )
    return frame.data, catalog


def save_candidates(image, catalog, output_directory, number=30, half_size=50):
    output_directory.mkdir(parents=True, exist_ok=True)
    candidates = catalog.head(number)
    height, width = image.shape
    for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
        x = int(round(row["x_centroid"]))
        y = int(round(row["y_centroid"]))
        x0 = max(0, x - half_size)
        x1 = min(width, x + half_size)
        y0 = max(0, y - half_size)
        y1 = min(height, y + half_size)
        cutout = image[y0:y1, x0:x1]
        finite = cutout[np.isfinite(cutout)]
        if finite.size == 0:
            continue
        vmin = np.percentile(finite, 5)
        vmax = np.percentile(finite, 99.5)
        ra = pd.to_numeric(row.get("ra", np.nan), errors="coerce")
        dec = pd.to_numeric(row.get("dec", np.nan), errors="coerce")
        snr = pd.to_numeric(row.get("local_snr", np.nan), errors="coerce")
        ra_text = f"{float(ra):.5f}" if np.isfinite(ra) else "null"
        dec_text = f"{float(dec):.5f}" if np.isfinite(dec) else "null"
        snr_text = f"{float(snr):.2f}" if np.isfinite(snr) else "null"
        status = row.get("match_status", "")
        plt.figure(figsize=(5, 5))
        plt.imshow(
            cutout,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(
            f"Candidate #{rank}\n"
            f"RA {ra_text}  Dec {dec_text}\n"
            f"{status}  SNR {snr_text}"
        )
        plt.tight_layout()
        filename = (
            output_directory
            / f"{rank:03d}_object_{int(row['label'])}.png"
        )
        plt.savefig(filename, dpi=160)
        plt.close()


def fetch_mast_exposure(filter_name, ra, dec, radius_deg, download_dir):
    print(
        "Querying MAST for one Roman exposure "
        f"({filter_name}, RA {ra}, Dec {dec}, radius {radius_deg} deg)..."
    )
    try:
        from astroquery.mast import Observations

        observations = Observations.query_criteria(
            coordinates=SkyCoord(ra, dec, unit="deg"),
            radius=radius_deg * u.deg,
            obs_collection="Roman",
        )
    except Exception as error:
        print(f"MAST query failed: {error}")
        return None
    if observations is None or len(observations) == 0:
        print("MAST returned no Roman observations.")
        return None
    try:
        products = Observations.get_product_list(observations)
        level2 = Observations.filter_products(
            products,
            calib_level=[2],
        )
        if level2 is None or len(level2) == 0:
            print("MAST returned no Level 2 products.")
            return None
        download_dir.mkdir(parents=True, exist_ok=True)
        manifest = Observations.download_products(
            level2[:1],
            download_dir=str(download_dir),
            flat=True,
        )
    except Exception as error:
        print(f"MAST download failed: {error}")
        return None
    if manifest is None or len(manifest) == 0:
        print("MAST download returned no files.")
        return None
    local_path = Path(str(manifest["Local Path"][0]))
    if not local_path.is_file():
        print("MAST download did not leave a local file.")
        return None
    print(f"Downloaded one MAST file: {local_path}")
    return local_path


def run_pipeline(
    input_path,
    top=30,
    reference=None,
    online=False,
    nsigma=4,
):
    input_path = Path(input_path)
    print()
    print("NGRST DISCOVERY TOOLKIT")
    print("=======================")
    print()
    print(f"Input: {input_path}")

    frame = load_image(input_path)
    print(f"Image dimensions: {frame.data.shape}")
    print(f"Filter: {frame.filter}")
    print(f"Exposure ID: {frame.exposure_id}")
    print(f"WCS found: {frame.wcs is not None}")

    print("1. Detecting astronomical sources...")
    image, catalog = _measure(frame, nsigma)
    print(f"   Sources detected: {len(catalog):,}")

    print("2. Assigning sky coordinates...")
    catalog = assign_sky(catalog, frame)
    catalog["filter"] = frame.filter
    catalog["exposure_id"] = frame.exposure_id

    print("3. Scoring morphology...")
    catalog = score_anomalies(catalog)

    if reference is None and DEFAULT_REFERENCE.is_file():
        reference = DEFAULT_REFERENCE
    reference_table = None
    if reference is not None:
        reference_path = Path(reference)
        if reference_path.is_file():
            print(f"4. Cross-matching locally: {reference_path}")
            reference_table = read_reference(reference_path)
        else:
            print(f"Local reference not found: {reference_path}")
    else:
        print("4. No local reference catalog.")
    catalog = crossmatch(catalog, reference_table)

    if online:
        catalog = crossmatch_online(catalog)
    else:
        print("Online cross-match skipped (no --crossmatch-online).")

    catalog = rank_catalog(catalog)
    catalog = _finalize(catalog)

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    catalog_path = PROCESSED_DIRECTORY / f"{input_path.stem}_catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)

    candidate_path = CANDIDATE_DIRECTORY / input_path.stem
    print("5. Rendering candidate images...")
    save_candidates(image, catalog, candidate_path, number=top)

    print()
    print("COMPLETE")
    print("========")
    print()
    print(f"Catalog:\n{catalog_path}")
    print()
    print(f"Candidate images:\n{candidate_path}")
    print()
    preview = [
        column
        for column in (
            "label",
            "ra",
            "dec",
            "match_status",
            "separation_arcsec",
            "local_snr",
            "anomaly_score",
        )
        if column in catalog.columns
    ]
    print("Ranked catalog:")
    if len(catalog):
        print(catalog[preview].head(10).to_string(index=False))
    else:
        print("(empty)")
    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="NGRST astronomical discovery toolkit"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .npz test data, FITS image, or Roman .asdf file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of candidate objects to export",
    )
    args = parser.parse_args()
    run_pipeline(args.input.resolve(), args.top)


if __name__ == "__main__":
    main()
