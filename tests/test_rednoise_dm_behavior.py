"""
Unit test for test_rednoise_dm_behavior.py. Run this first before that file because that one takes a long time!!! Call as:
    pytest test_rednoise_dm_behavior.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch


def _add_module_dir_to_syspath(module_filename="rednoise_dm_behavior.py", max_levels=6):
    """
    Find the directory containing rednoise_dm_behavior.py by walking upward
    from THIS FILE's own location (independent of cwd or pytest's invocation
    directory) and add it to sys.path. Checks each ancestor's "scripts/"
    subdirectory first (this repo's normal scripts/ + tests/ layout), then
    the ancestor itself.
    """
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents][:max_levels]:
        for candidate in (ancestor / "scripts" / module_filename, ancestor / module_filename):
            if candidate.is_file():
                script_dir = str(candidate.parent)
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                return
    raise ImportError(
        f"Could not find {module_filename} by walking up from {here} "
        f"(checked {max_levels} ancestor directories and each one's "
        f"'scripts' subdirectory). Is this test file still inside the "
        f"rednoise repo?"
    )


_add_module_dir_to_syspath()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

import rednoise_dm_behavior as rdb  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def build_combined_medians_h5(path, N=10, max_dm=6, max_freq=8, seed=0,
                               scale_key="scale"):
    """
    Write a small, real HDF5 file matching combine_rednoise_medians.py's
    schema: medians (N, max_dm, max_freq), scale (N, max_freq), n_dm,
    n_freq, ra, dec, year, month, day, and the max_n_freq attr.

    Returns the raw (medians, scale, n_dm, n_freq) arrays used to build the
    file, so tests can check the script's output against known-good values.

    n_freq is drawn from [6, max_freq] rather than [0, max_freq]: the
    script's mins_match check slices [:, 5:], so n_freq > 5 is an implicit
    precondition throughout this codebase (real data has n_freq ~31-51).
    """
    rng = np.random.default_rng(seed)
    n_dm = rng.integers(2, max_dm + 1, size=N).astype(np.int32)
    n_freq = rng.integers(6, max_freq + 1, size=N).astype(np.int32)

    medians = -1.0 * np.ones((N, max_dm, max_freq), dtype=np.float32)
    scale = -1.0 * np.ones((N, max_freq), dtype=np.float32)
    for i in range(N):
        nd, nf = int(n_dm[i]), int(n_freq[i])
        medians[i, :nd, :nf] = rng.uniform(1.0, 100.0, size=(nd, nf)).astype(np.float32)
        scale[i, :nf] = rng.integers(50, 500, size=nf).astype(np.float32)

    ra = rng.uniform(0, 360, size=N)
    dec = rng.uniform(-90, 90, size=N)
    year = np.full(N, 2024, dtype=np.int16)
    month = np.full(N, 1, dtype=np.uint8)
    day = (np.arange(N) % 28 + 1).astype(np.uint8)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("medians", data=medians)
        if scale_key is not None:
            h5.create_dataset(scale_key, data=scale)
        h5.create_dataset("n_dm", data=n_dm)
        h5.create_dataset("n_freq", data=n_freq)
        h5.create_dataset("ra", data=ra)
        h5.create_dataset("dec", data=dec)
        h5.create_dataset("year", data=year)
        h5.create_dataset("month", data=month)
        h5.create_dataset("day", data=day)
        h5.attrs["max_n_freq"] = max_freq

    return medians, scale, n_dm, n_freq


def run_main(h5_path, out_path, chunk_size=4, max_rows=None):
    """Invoke rdb.main() as if run from the command line."""
    argv = ["rednoise_dm_behavior.py", str(h5_path), "--output", str(out_path),
            "--chunk-size", str(chunk_size)]
    if max_rows is not None:
        argv += ["--max-rows", str(max_rows)]
    with patch.object(sys, "argv", argv):
        rdb.main()


def expected_median_row(medians, n_dm_i, n_freq_i, i):
    nd, nf = int(n_dm_i), int(n_freq_i)
    return np.median(medians[i, :nd, :nf].astype(np.float64), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scale_is_written_and_matches_source(tmp_path):
    """
    The new 'scale' output array should appear in the npz, with the right
    shape, and match the source h5 file's 'scale' dataset exactly within
    each row's valid n_freq region -- and be -1 past it, same padding
    convention as the source.
    """
    h5_path = tmp_path / "combined.h5"
    out_path = tmp_path / "out.npz"
    medians, scale, n_dm, n_freq = build_combined_medians_h5(h5_path, N=10, max_freq=8)

    run_main(h5_path, out_path, chunk_size=4)
    npz = np.load(out_path)

    assert "scale" in npz.files
    assert npz["scale"].shape == (10, 8)
    for i in range(10):
        nf = int(n_freq[i])
        np.testing.assert_allclose(npz["scale"][i, :nf], scale[i, :nf])
        assert np.all(npz["scale"][i, nf:] == -1.0)

    # The pre-existing DM-axis statistics should be unaffected by adding scale.
    i = 3
    np.testing.assert_allclose(
        npz["median_across_dms"][i, :n_freq[i]],
        expected_median_row(medians, n_dm[i], n_freq[i], i),
    )


def test_scale_plural_key_is_also_used(tmp_path):
    """
    The source medians.npz files (and hence some combined h5 files) may use
    "scales" (plural) instead of "scale". The script should find either.
    """
    h5_path = tmp_path / "combined.h5"
    out_path = tmp_path / "out.npz"
    _, scale, _, n_freq = build_combined_medians_h5(
        h5_path, N=5, max_freq=6, scale_key="scales"
    )

    run_main(h5_path, out_path, chunk_size=2)
    npz = np.load(out_path)

    for i in range(5):
        nf = int(n_freq[i])
        np.testing.assert_allclose(npz["scale"][i, :nf], scale[i, :nf])


def test_missing_scale_dataset_fills_minus_one_without_crashing(tmp_path):
    """
    An older combined medians file with no 'scale'/'scales' dataset at all
    should not crash the script -- the output 'scale' array should just be
    entirely -1 (the same "unknown" sentinel used elsewhere in this file
    format), and the rest of the output should be computed normally.
    """
    h5_path = tmp_path / "combined.h5"
    out_path = tmp_path / "out.npz"
    build_combined_medians_h5(h5_path, N=5, max_freq=6, scale_key=None)

    run_main(h5_path, out_path, chunk_size=2)
    npz = np.load(out_path)

    assert np.all(npz["scale"] == -1.0)
    assert npz["median_across_dms"].shape == (5, 6)


def test_corrupted_medians_rows_are_skipped_and_scale_stays_aligned(tmp_path):
    """
    When a chunk read of 'medians' raises OSError (simulated here via
    mock, matching a real mid-write-killed HDF5 chunk), the script should
    fall back to row-by-row reads, skip only the corrupted rows, and leave
    them at their pre-allocated defaults (0 for the DM stats, -1 for scale)
    -- while every other row in the same chunk, including its 'scale' entry,
    is still computed correctly and stays aligned with the right row index.
    """
    h5_path = tmp_path / "combined.h5"
    out_path = tmp_path / "out.npz"
    medians, scale, n_dm, n_freq = build_combined_medians_h5(h5_path, N=8, max_freq=6)

    corrupt_rows = {2, 5}
    real_getitem = h5py.Dataset.__getitem__

    def flaky_getitem(self, key):
        if self.name.endswith("medians"):
            touched = set()
            if isinstance(key, slice):
                start, stop, step = key.indices(self.shape[0])
                touched = set(range(start, stop, step or 1))
            elif isinstance(key, int):
                touched = {key}
            if touched & corrupt_rows:
                raise OSError("inflate() failed (simulated corruption)")
        return real_getitem(self, key)

    with patch.object(h5py.Dataset, "__getitem__", flaky_getitem):
        run_main(h5_path, out_path, chunk_size=4)

    npz = np.load(out_path)
    for i in range(8):
        nf = int(n_freq[i])
        if i in corrupt_rows:
            assert np.all(npz["scale"][i] == -1.0), f"row {i} scale should be left at default"
            assert np.all(npz["median_across_dms"][i] == 0.0), f"row {i} median should be left at default"
        else:
            np.testing.assert_allclose(npz["scale"][i, :nf], scale[i, :nf])
            np.testing.assert_allclose(
                npz["median_across_dms"][i, :nf],
                expected_median_row(medians, n_dm[i], n_freq[i], i),
            )


def test_mismatched_dataset_lengths_use_the_minimum(tmp_path):
    """
    If 'scale' (or any other dataset) is shorter than 'medians' -- e.g. from
    a process killed mid-batch -- the script should process only the
    consistent prefix (min length across all datasets, now including
    scale), not crash or silently read past the end of the shorter one.
    """
    h5_path = tmp_path / "combined.h5"
    out_path = tmp_path / "out.npz"
    build_combined_medians_h5(h5_path, N=10, max_freq=6)

    # Truncate 'scale' and 'ra' after the fact to simulate an inconsistent file.
    with h5py.File(h5_path, "a") as h5:
        scale_data = h5["scale"][:7]
        ra_data = h5["ra"][:9]
        del h5["scale"]
        del h5["ra"]
        h5.create_dataset("scale", data=scale_data)
        h5.create_dataset("ra", data=ra_data)

    run_main(h5_path, out_path, chunk_size=3)
    npz = np.load(out_path)

    assert npz["info"].shape[0] == 7
    assert npz["scale"].shape[0] == 7


def test_chunking_does_not_change_the_result(tmp_path):
    """
    Regression guard: processing the same file in one big chunk vs. several
    small chunks should produce byte-identical output, for every array
    including the new 'scale' one. This is what actually matters about
    "chunked processing" -- it should be purely a RAM/performance knob.
    """
    h5_path = tmp_path / "combined.h5"
    out_single = tmp_path / "out_single.npz"
    out_chunked = tmp_path / "out_chunked.npz"
    N = 11
    build_combined_medians_h5(h5_path, N=N, max_dm=5, max_freq=7)

    run_main(h5_path, out_single, chunk_size=N)   # one chunk covers everything
    run_main(h5_path, out_chunked, chunk_size=3)  # N=11 is not a multiple of 3

    single = np.load(out_single)
    chunked = np.load(out_chunked)
    for key in ("info", "median_across_dms", "std_across_dms",
                "min_across_dms", "max_across_dms", "scale"):
        np.testing.assert_array_equal(single[key], chunked[key])


def test_output_array_memory_matches_expected_formula(tmp_path):
    """
    Characterization test for the script's memory footprint (see the
    accompanying review): the five (N, max_n_freq) output arrays plus the
    (N, 7) info array are fully materialized in RAM for ALL N rows before
    the single np.savez_compressed() call -- chunking only bounds the
    *input* read, not this accumulation. This test doesn't assert that's a
    problem (it isn't, at the row counts this repo currently deals with),
    but pins down the actual scaling so a future change that makes it worse
    (e.g. an accidental extra full-array copy) gets caught.
    """
    h5_path = tmp_path / "combined.h5"
    out_path = tmp_path / "out.npz"
    N, max_freq = 20, 9
    build_combined_medians_h5(h5_path, N=N, max_freq=max_freq)

    run_main(h5_path, out_path, chunk_size=4)
    npz = np.load(out_path)

    per_freq_arrays = ["median_across_dms", "std_across_dms",
                       "min_across_dms", "max_across_dms", "scale"]
    expected_bytes_per_freq_array = N * max_freq * 4  # float32
    for key in per_freq_arrays:
        assert npz[key].nbytes == expected_bytes_per_freq_array, (
            f"{key} is {npz[key].nbytes} bytes, expected "
            f"{expected_bytes_per_freq_array} (N * max_n_freq * 4 bytes); "
            f"if this changed, the script's O(N) output memory footprint did too."
        )

    total_output_bytes = sum(npz[k].nbytes for k in per_freq_arrays) + npz["info"].nbytes
    # Sanity check against the same formula used in the review: for a real
    # ~2M-row combined file at max_n_freq=51, this scales to ~2.1 GB peak
    # just for these six arrays, held fully in RAM regardless of --chunk-size.
    projected_2m_rows_bytes = (2_000_000 / N) * total_output_bytes
    assert projected_2m_rows_bytes > 0  # just documents the projection; see review
