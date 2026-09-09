#!/usr/bin/env python3
"""
test_combine_rednoise_medians.py
=================================
Unit tests for combine_rednoise_medians.py.

These tests build small, synthetic medians.npz trees under pytest's tmp_path,
run the real combine() function against them, then open the resulting HDF5
output and check that every dataset and attribute it should carry is present,
correctly shaped, and correctly valued -- in particular the "scales" ->
"scale" field this script was extended to save.

Run with:
    pip install pytest numpy h5py tqdm
    pytest test_combine_rednoise_medians.py -v

Layout assumption: this file lives in a "tests" directory, and
combine_rednoise_medians.py lives in a sibling "scripts" directory one level
up, i.e.:
    <repo>/scripts/combine_rednoise_medians.py
    <repo>/tests/test_combine_rednoise_medians.py   (this file)
"""

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the script under test as a module (it's a script, not a package).
# It lives in ../scripts relative to this test file's directory.
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "combine_rednoise_medians.py"
if not _SCRIPT_PATH.exists():
    raise FileNotFoundError(
        f"Could not find combine_rednoise_medians.py at {_SCRIPT_PATH}. "
        "Expected it in a 'scripts' directory one level up from this test's "
        "'tests' directory (<repo>/scripts/combine_rednoise_medians.py)."
    )

_spec = importlib.util.spec_from_file_location("combine_rednoise_medians", _SCRIPT_PATH)
combine_mod = importlib.util.module_from_spec(_spec)
sys.modules["combine_rednoise_medians"] = combine_mod
_spec.loader.exec_module(combine_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pointing(root: Path, year: int, month: int, day: int,
                   ra: float, dec: float, n_dm: int, n_freq: int,
                   scale_ndim: int | None = 1, scale_key: str = "scales",
                   seed: int = 0):
    """
    Write one synthetic {root}/{year}/{month:02d}/{day:02d}/{ra}_{dec}/medians.npz.

    scale_ndim: 1, 2, or 3 -- controls how the 'scales' array is shaped
        before being written (mirroring the real files, where the per-file
        scale can come as a 1D, 2D, or 3D array whose extra axes are all
        identical copies). None omits the scale key entirely.
    scale_key: the npz key to store the scale array under -- "scales" is
        what the real files use; "scale" (singular) exercises the fallback.

    Returns (expected_medians, expected_scale_1d) so tests can assert
    against known-good values. expected_scale_1d is None if scale_ndim is None.
    """
    rng = np.random.default_rng(seed)
    pdir = root / str(year) / f"{month:02d}" / f"{day:02d}" / f"{ra}_{dec}"
    pdir.mkdir(parents=True, exist_ok=True)

    medians = rng.random((1, n_dm, n_freq)).astype(np.float32)

    save_kwargs = {"medians": medians}
    expected_scale = None
    if scale_ndim is not None:
        scale_1d = (np.arange(n_freq, dtype=np.float32) + 1) * 0.1
        expected_scale = scale_1d
        if scale_ndim == 1:
            scale_arr = scale_1d
        elif scale_ndim == 2:
            scale_arr = np.tile(scale_1d, (n_dm, 1))
        elif scale_ndim == 3:
            scale_arr = np.tile(scale_1d, (1, n_dm, 1))
        else:
            raise ValueError(f"scale_ndim must be 1, 2, or 3; got {scale_ndim}")
        save_kwargs[scale_key] = scale_arr

    np.savez(pdir / "medians.npz", **save_kwargs)
    return medians[0], expected_scale


def index_by_radec(h5: h5py.File, ra: float, dec: float) -> int:
    """Find the row index for a given (ra, dec) pair in an open output file."""
    ras = h5["ra"][:]
    decs = h5["dec"][:]
    matches = np.nonzero((ras == ra) & (decs == dec))[0]
    assert len(matches) == 1, f"expected exactly one row for ({ra}, {dec}), found {len(matches)}"
    return int(matches[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rednoise_root(tmp_path):
    return tmp_path / "rednoise_root"


@pytest.fixture
def output_h5(tmp_path):
    return tmp_path / "combined.h5"


# ---------------------------------------------------------------------------
# Tests: basic structure -- the file opens, and every documented dataset
# and attribute is present with the right dtype/shape.
# ---------------------------------------------------------------------------

class TestOutputFileStructure:

    def test_output_file_opens_and_has_expected_datasets(self, rednoise_root, output_h5):
        make_pointing(rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=5, n_freq=8)
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=6, n_freq=10)

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        assert output_h5.exists()
        with h5py.File(output_h5, "r") as f:
            expected_datasets = {
                "medians", "scale", "n_dm", "n_freq",
                "ra", "dec", "year", "month", "day", "_done",
            }
            assert expected_datasets.issubset(set(f.keys())), (
                f"missing datasets: {expected_datasets - set(f.keys())}"
            )

    def test_dataset_shapes_and_dtypes(self, rednoise_root, output_h5):
        make_pointing(rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=5, n_freq=8)
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=6, n_freq=10)

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            n = f["medians"].shape[0]
            assert n == 2

            # medians is (N, n_dm_max, n_freq_max); scale is (N, n_freq_max)
            assert f["medians"].shape == (n, 6, 10)
            assert f["scale"].shape == (n, 10)
            assert f["medians"].dtype == np.float32
            assert f["scale"].dtype == np.float32

            for name, dtype in [
                ("n_dm", np.int32), ("n_freq", np.int32),
                ("ra", np.float64), ("dec", np.float64),
                ("year", np.int16), ("month", np.uint8), ("day", np.uint8),
            ]:
                assert f[name].shape == (n,)
                assert f[name].dtype == dtype

    def test_attrs_present_and_correct(self, rednoise_root, output_h5):
        make_pointing(rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=5, n_freq=8)
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=6, n_freq=10)
        make_pointing(rednoise_root, 2024, 1, 3, 12.0, 22.0, n_dm=4, n_freq=7)

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            for attr in ("max_n_dm", "max_n_freq", "pad_sentinel", "total_expected"):
                assert attr in f.attrs, f"missing attribute: {attr}"

            assert int(f.attrs["max_n_dm"]) == 6
            assert int(f.attrs["max_n_freq"]) == 10
            assert float(f.attrs["pad_sentinel"]) == -1.0
            assert int(f.attrs["total_expected"]) == 3


# ---------------------------------------------------------------------------
# Tests: the 'scale' field itself -- correct reduction from 1D/2D/3D source
# arrays, correct key ('scales' primary, 'scale' fallback), correct
# behaviour when the key is missing entirely.
# ---------------------------------------------------------------------------

class TestScaleField:

    @pytest.mark.parametrize("scale_ndim", [1, 2, 3])
    def test_scale_reduced_correctly_for_each_dimensionality(
        self, rednoise_root, output_h5, scale_ndim,
    ):
        _, expected_scale = make_pointing(
            rednoise_root, 2024, 1, 1, 10.0, 20.0,
            n_dm=5, n_freq=8, scale_ndim=scale_ndim, scale_key="scales",
        )

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            i = index_by_radec(f, 10.0, 20.0)
            n_freq = f["n_freq"][i]
            got_scale = f["scale"][i, :n_freq]
            np.testing.assert_allclose(got_scale, expected_scale)

    def test_scale_key_fallback_singular(self, rednoise_root, output_h5):
        """Older/other files may use 'scale' (singular) -- should still work."""
        _, expected_scale = make_pointing(
            rednoise_root, 2024, 1, 1, 10.0, 20.0,
            n_dm=5, n_freq=8, scale_ndim=1, scale_key="scale",
        )

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            i = index_by_radec(f, 10.0, 20.0)
            n_freq = f["n_freq"][i]
            got_scale = f["scale"][i, :n_freq]
            np.testing.assert_allclose(got_scale, expected_scale)

    def test_missing_scale_key_stores_sentinel(self, rednoise_root, output_h5):
        make_pointing(
            rednoise_root, 2024, 1, 1, 10.0, 20.0,
            n_dm=5, n_freq=8, scale_ndim=None,
        )

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            i = index_by_radec(f, 10.0, 20.0)
            n_freq = f["n_freq"][i]
            got_scale = f["scale"][i, :n_freq]
            assert np.all(got_scale == -1.0)

    def test_scale_padding_beyond_n_freq_is_sentinel(self, rednoise_root, output_h5):
        # Two pointings with different n_freq -- the shorter one's row must
        # be padded with -1 beyond its own n_freq.
        make_pointing(rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=5, n_freq=6)
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=5, n_freq=10)

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            i = index_by_radec(f, 10.0, 20.0)
            n_freq = f["n_freq"][i]
            assert n_freq == 6
            row = f["scale"][i]
            assert row.shape[0] == 10  # padded to global max_freq
            assert np.all(row[n_freq:] == -1.0)
            assert np.all(row[:n_freq] != -1.0)


# ---------------------------------------------------------------------------
# Tests: n_dm[i]/n_freq[i] correctly slice out the valid (non-padded) region
# of medians[i] and scale[i], matching the documented usage pattern.
# ---------------------------------------------------------------------------

class TestSlicingConvention:

    def test_medians_and_scale_slice_to_exact_original_data(self, rednoise_root, output_h5):
        expected_medians, expected_scale = make_pointing(
            rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=4, n_freq=6,
        )
        # a second, larger pointing forces padding on the first one
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=9, n_freq=13)

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            i = index_by_radec(f, 10.0, 20.0)
            m = f["medians"][i, :f["n_dm"][i], :f["n_freq"][i]]
            s = f["scale"][i, :f["n_freq"][i]]
            np.testing.assert_allclose(m, expected_medians)
            np.testing.assert_allclose(s, expected_scale)


# ---------------------------------------------------------------------------
# Tests: resume behaviour, including the shape-growth path (a resumed run
# hitting a file bigger than anything the first run saw).
# ---------------------------------------------------------------------------

class TestResume:

    def test_resume_skips_already_written_files(self, rednoise_root, output_h5):
        make_pointing(rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=5, n_freq=6)
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=5, n_freq=6)

        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)
        with h5py.File(output_h5, "r") as f:
            assert f["medians"].shape[0] == 2

        # add a third pointing and re-run against the same output file
        make_pointing(rednoise_root, 2024, 1, 3, 12.0, 22.0, n_dm=5, n_freq=6)
        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            assert f["medians"].shape[0] == 3
            assert len(f["_done"]) == 3

    def test_resume_grows_padded_shape_for_larger_later_file(self, rednoise_root, output_h5):
        # First run only sees small-shape pointings.
        make_pointing(rednoise_root, 2024, 1, 1, 10.0, 20.0, n_dm=5, n_freq=6)
        make_pointing(rednoise_root, 2024, 1, 2, 11.0, 21.0, n_dm=5, n_freq=6)
        combine_mod.combine(
            rednoise_root, output_h5, batch_size=500, workers=4, n_files=2,
        )
        with h5py.File(output_h5, "r") as f:
            assert f["medians"].shape == (2, 5, 6)
            assert int(f.attrs["max_n_dm"]) == 5
            assert int(f.attrs["max_n_freq"]) == 6

        # A third, larger pointing appears; resuming must not crash and must
        # grow the padded shape (with correct -1 backfill) instead.
        expected_medians, expected_scale = make_pointing(
            rednoise_root, 2024, 1, 3, 12.0, 22.0, n_dm=9, n_freq=11,
        )
        combine_mod.combine(rednoise_root, output_h5, batch_size=500, workers=4)

        with h5py.File(output_h5, "r") as f:
            assert f["medians"].shape[0] == 3
            assert f["medians"].shape[1:] == (9, 11)
            assert f["scale"].shape[1:] == (11,)
            assert int(f.attrs["max_n_dm"]) == 9
            assert int(f.attrs["max_n_freq"]) == 11

            # the two originally-small rows must be backfilled with -1 in
            # their newly grown region, not garbage/zeros
            for ra, dec in [(10.0, 20.0), (11.0, 21.0)]:
                i = index_by_radec(f, ra, dec)
                assert np.all(f["medians"][i, 5:, :] == -1.0)
                assert np.all(f["medians"][i, :, 6:] == -1.0)
                assert np.all(f["scale"][i, 6:] == -1.0)

            # the new, larger row's real data must be intact and unpadded
            i = index_by_radec(f, 12.0, 22.0)
            m = f["medians"][i, :f["n_dm"][i], :f["n_freq"][i]]
            s = f["scale"][i, :f["n_freq"][i]]
            np.testing.assert_allclose(m, expected_medians)
            np.testing.assert_allclose(s, expected_scale)


# ---------------------------------------------------------------------------
# Tests: --n-files / n_files limiting.
# ---------------------------------------------------------------------------

class TestNFilesLimit:

    def test_n_files_limits_total_pointings_written(self, rednoise_root, output_h5):
        for i in range(5):
            make_pointing(rednoise_root, 2024, 1, i + 1, 10.0 + i, 20.0 + i, n_dm=4, n_freq=5)

        combine_mod.combine(
            rednoise_root, output_h5, batch_size=500, workers=4, n_files=3,
        )

        with h5py.File(output_h5, "r") as f:
            assert f["medians"].shape[0] == 3
            assert int(f.attrs["total_expected"]) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
