import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import ngrst
import roman_toolkit


FIXTURE = ROOT / "tests" / "fixtures" / "detector_fixture.asdf"
TRUTH = ROOT / "tests" / "fixtures" / "truth.ecsv"


def _sky_separation_arcsec(ra1, dec1, ra2, dec2):
    left = SkyCoord(np.asarray(ra1) * u.deg, np.asarray(dec1) * u.deg)
    right = SkyCoord(np.asarray(ra2) * u.deg, np.asarray(dec2) * u.deg)
    return left.separation(right).arcsec


def test_fixture_frame_has_roman_metadata():
    frame = roman_toolkit.load_image(FIXTURE)
    assert frame.source == "asdf"
    assert frame.wcs is not None
    assert frame.filter == "F158"
    assert frame.exposure_id == "FIXTURE-001"
    assert frame.err is not None
    assert frame.data.shape == (256, 256)
    assert frame.dq.dtype == np.uint32
    assert int(frame.dq.sum()) > 0


def test_sky_catalog_marks_planted_sources(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    candidates = tmp_path / "candidates"
    monkeypatch.setattr(roman_toolkit, "PROCESSED_DIRECTORY", processed)
    monkeypatch.setattr(roman_toolkit, "CANDIDATE_DIRECTORY", candidates)

    catalog = roman_toolkit.run_pipeline(FIXTURE, top=8)
    truth = Table.read(TRUTH).to_pandas()
    unknowns = truth[truth["kind"] == "unknown"]
    knowns = truth[truth["kind"] == "known"]
    dq = truth[truth["kind"] == "dq"].iloc[0]

    dq_distance = np.hypot(
        catalog["x_centroid"] - dq["x"],
        catalog["y_centroid"] - dq["y"],
    )
    assert not bool((dq_distance < 8).any())

    top = catalog.head(8)
    assert top["match_status"].eq("unknown").all()
    for row in unknowns.itertuples(index=False):
        separation = _sky_separation_arcsec(
            catalog["ra"],
            catalog["dec"],
            row.ra,
            row.dec,
        )
        nearest = int(np.argmin(separation))
        assert separation[nearest] < 0.2
        assert catalog.loc[nearest, "match_status"] == "unknown"
        assert nearest < 8

    for row in knowns.itertuples(index=False):
        separation = _sky_separation_arcsec(
            catalog["ra"],
            catalog["dec"],
            row.ra,
            row.dec,
        )
        nearest = int(np.argmin(separation))
        assert separation[nearest] < 0.2
        assert catalog.loc[nearest, "match_status"] == "known"
        assert catalog.loc[nearest, "ref_id"] == row.source_id

    assert catalog["filter"].eq("F158").all()
    assert catalog["exposure_id"].eq("FIXTURE-001").all()
    assert catalog["detect_status"].eq("error_array").all()
    written = pd.read_parquet(processed / "detector_fixture_catalog.parquet")
    assert list(written.columns[: len(roman_toolkit.CATALOG_COLUMNS)]) == (
        roman_toolkit.CATALOG_COLUMNS
    )
    assert len(list((candidates / "detector_fixture").glob("*.png"))) == 8


def test_discover_command_skips_network():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ngrst",
            "discover",
            str(FIXTURE),
            "--top",
            "8",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "MAST query skipped (no --mast)." in completed.stdout
    assert "Online cross-match skipped (no --crossmatch-online)." in completed.stdout
    assert "WCS found: True" in completed.stdout
    catalog = pd.read_parquet(ROOT / "data" / "processed" / "detector_fixture_catalog.parquet")
    assert catalog.head(4)["match_status"].eq("unknown").all()
    assert catalog.head(4)["ra"].notna().all()
    assert catalog.head(4)["dec"].notna().all()


def test_mast_flag_exits_when_archive_is_empty(monkeypatch, capsys):
    monkeypatch.setattr(roman_toolkit, "fetch_mast_exposure", lambda **kwargs: None)
    code = ngrst.main(
        [
            "discover",
            "--mast",
            "--filter",
            "F158",
            "--ra",
            "150.0",
            "--dec",
            "2.0",
            "--radius",
            "0.05",
        ]
    )
    assert code == 0
    assert "Nothing to discover." in capsys.readouterr().out


def test_online_crossmatch_failure_keeps_local_status(monkeypatch):
    def boom(ra, dec, radius_arcsec):
        raise RuntimeError("offline")

    monkeypatch.setattr(roman_toolkit, "query_simbad_region", boom)
    catalog = pd.DataFrame(
        {
            "coord_status": ["sky"],
            "ra": [150.0],
            "dec": [2.0],
            "match_status": ["unknown"],
            "ref_id": [None],
        }
    )
    updated = roman_toolkit.crossmatch_online(catalog)
    assert updated.loc[0, "match_status"] == "unknown"


def test_fits_and_npz_wcs_roundtrip(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = [8.5, 8.5]
    wcs.wcs.crval = [10.0, -5.0]
    wcs.wcs.cdelt = [-0.001, 0.001]
    image = np.zeros((16, 16), dtype=np.float64)
    header = wcs.to_header()
    header["FILTER"] = "F158"
    header["OBS_ID"] = "FITS-1"
    fits_path = tmp_path / "scene.fits"
    fits.PrimaryHDU(data=image, header=header).writeto(fits_path)
    fits_frame = roman_toolkit.load_image(fits_path)
    assert fits_frame.source == "fits"
    assert fits_frame.wcs is not None
    assert fits_frame.filter == "F158"
    assert fits_frame.exposure_id == "FITS-1"
    ra, dec = fits_frame.wcs.pixel_to_world_values(7.5, 7.5)
    assert abs(float(np.asarray(ra).reshape(-1)[0]) - 10.0) < 1e-6
    assert abs(float(np.asarray(dec).reshape(-1)[0]) + 5.0) < 1e-6

    npz_path = tmp_path / "scene.npz"
    np.savez(
        npz_path,
        data=image,
        dq=np.zeros(image.shape, dtype=np.uint32),
        err=np.ones(image.shape),
        wcs=np.array(header.tostring(sep="\n")),
        filter=np.array("F158"),
        exposure_id=np.array("NPZ-1"),
    )
    npz_frame = roman_toolkit.load_image(npz_path)
    assert npz_frame.source == "npz"
    assert npz_frame.err is not None
    assert npz_frame.wcs is not None
    assert npz_frame.filter == "F158"
    assert npz_frame.exposure_id == "NPZ-1"
