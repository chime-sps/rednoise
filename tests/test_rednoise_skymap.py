"""
Unit test for rednoise_skymap.py. Call as:
    pytest test_rednoise_skymap.py -v
"""
import sys
from pathlib import Path


def _add_rednoise_skymap_dir_to_syspath(module_filename="rednoise_skymap.py", max_levels=6):
    """
    Find the directory containing rednoise_skymap.py by walking upward from
    THIS FILE's own location (Path(__file__), which is independent of the
    current working directory or pytest's invocation directory) and add it
    to sys.path. Checked at each ancestor level: a "scripts/<module_filename>"
    subdirectory first (this repo's normal scripts/ + tests/ layout), then
    the ancestor directory itself, in case the layout differs.
    """
    here = Path(__file__).resolve().parent
    ancestors = [here, *here.parents][:max_levels]

    for ancestor in ancestors:
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


def _stub_h5py_if_missing():
    """
    Install a lightweight stand-in for h5py, only if the real package isn't
    installed. rednoise_skymap.py does `import h5py` at module scope (for
    load_exposure_lookup()), but grid_and_smooth() -- the function under
    test here -- never touches it.
    """
    import importlib
    try:
        importlib.import_module("h5py")
        return
    except ImportError:
        pass

    import types
    mod = types.ModuleType("h5py")

    def _no_file(*args, **kwargs):
        raise RuntimeError(
            "h5py stub active: this test suite exercises pure "
            "gridding/smoothing logic and never opens a real HDF5 file."
        )

    mod.File = _no_file
    sys.modules["h5py"] = mod


def _stub_chime_sps_if_missing():
    """
    Install a lightweight stand-in for chime_sps.sps_constants, only if the
    real package isn't installed. rednoise_skymap.py does
    `from chime_sps.sps_constants import tsamp` at module scope, but
    grid_and_smooth() never touches tsamp.

    Both the parent package ("chime_sps") and the submodule
    ("chime_sps.sps_constants") must be registered in sys.modules -- Python
    resolves "from chime_sps.sps_constants import tsamp" by importing
    "chime_sps" first, so registering only the submodule isn't enough.
    """
    import importlib
    try:
        importlib.import_module("chime_sps.sps_constants")
        return
    except ImportError:
        pass

    import types
    pkg = types.ModuleType("chime_sps")
    const_mod = types.ModuleType("chime_sps.sps_constants")
    # Placeholder only -- these tests don't depend on the real tsamp value.
    const_mod.tsamp = 0.00098304
    pkg.sps_constants = const_mod
    sys.modules["chime_sps"] = pkg
    sys.modules["chime_sps.sps_constants"] = const_mod


_add_rednoise_skymap_dir_to_syspath()
_stub_h5py_if_missing()
_stub_chime_sps_if_missing()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rednoise_skymap import grid_and_smooth  # noqa: E402


def _smoothed_value_near(lon_grid, dec_grid, smoothed, ra_query, dec_query):
    """Return the smoothed grid value at the cell nearest (ra_query, dec_query)."""
    lon_query = -(ra_query % 360.0)
    lon_query = (lon_query + 180.0) % 360.0 - 180.0
    lon_idx = np.argmin(np.abs(lon_grid - lon_query))
    dec_idx = np.argmin(np.abs(dec_grid - dec_query))
    return smoothed[dec_idx, lon_idx]


def test_flat_field_recovers_exact_value_away_from_edges():
    """
    Control/sanity check: a spatially uniform field, gridded and smoothed
    somewhere with no seam and no pole nearby, should come back out exactly
    uniform. This isolates the artifacts below from ordinary, expected
    smoothing/coverage-normalization behaviour.
    """
    rng = np.random.default_rng(0)
    n = 200
    ra = rng.uniform(60.0, 120.0, n)    # well clear of the RA=180 seam
    dec = rng.uniform(-20.0, 20.0, n)   # well clear of the poles
    value = np.full(n, 42.0)

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, value, smooth_deg=3.0, grid_resolution=0.5
    )

    finite = smoothed[np.isfinite(smoothed)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 42.0, atol=1e-6)


def test_ra_180_seam_preserves_contrast_that_should_be_smoothed_out():
    """
    A hot point and a cold point 2 degrees apart should be blended by
    Gaussian smoothing (sigma = 5 deg) by roughly the same amount no matter
    *where* on the sky that pair sits. Here the identical pair is placed
    once away from any wrap boundary (control) and once straddling RA=180
    (seam). With periodic longitude smoothing, the seam pair should now be
    blended together just as well as the control pair, since 179 deg and
    181 deg are only 2 degrees apart circularly, same as 89 deg and 91 deg.
    """
    grid_res = 1.0
    smooth_deg = 5.0
    hot_value = 1000.0
    cold_value = 0.0

    # Control pair: 2 degrees apart, far from any wrap boundary.
    ra_control = np.array([89.0, 91.0])
    dec_control = np.array([0.0, 0.0])
    values_control = np.array([hot_value, cold_value])

    # Seam pair: identical geometry (2 degrees apart, same Dec), straddling
    # RA=180, where -(ra % 360) wrapped to (-180, 180] jumps from just
    # under +180 to just over -180.
    ra_seam = np.array([179.0, 181.0])
    dec_seam = np.array([0.0, 0.0])
    values_seam = np.array([hot_value, cold_value])

    lon_c, dec_c, smoothed_c = grid_and_smooth(
        ra_control, dec_control, values_control, smooth_deg, grid_resolution=grid_res
    )
    lon_s, dec_s, smoothed_s = grid_and_smooth(
        ra_seam, dec_seam, values_seam, smooth_deg, grid_resolution=grid_res
    )

    peak_control = _smoothed_value_near(lon_c, dec_c, smoothed_c, 89.0, 0.0)
    peak_seam = _smoothed_value_near(lon_s, dec_s, smoothed_s, 179.0, 0.0)

    # With RA handled as periodic, the hot point's smoothed peak should be
    # attenuated by essentially the same amount in both cases -- in fact
    # exactly the same, since the two configurations are related by a pure
    # rotation in longitude once wrap-around is respected.
    assert peak_seam == pytest.approx(peak_control, abs=1.0), (
        f"Hot point peak near the RA=180 seam ({peak_seam:.1f}) retained "
        f"more of its raw contrast than the seam-free control "
        f"({peak_control:.1f}); the projection isn't treating RA as "
        f"periodic, artificially inflating dynamic range at the seam."
    )


def test_high_declination_kernel_covers_larger_true_sky_area():
    """
    Two point sources at the same *true* angular separation should be
    smoothed together to a similar degree regardless of declination. The
    polar pair here is 36 raw RA degrees apart, but at Dec=89 that's
    36 * cos(89 deg) =~ 0.63 true degrees -- essentially the same true
    separation as the equatorial pair, placed 0.6 degrees apart in raw RA.
    With the longitude smoothing sigma scaled by 1/cos(dec), both pairs
    should now be blended by a similar amount, unlike the pre-fix behaviour
    (sigma fixed in raw RA degrees), which barely touched the polar pair
    (36 raw degrees is far outside a 5 deg kernel) while heavily blending
    the equatorial one (0.6 deg is well inside it). The two-pass separable
    smoothing here (per-row longitude pass, then a declination pass) is
    still only an approximation of a true 2D isotropic kernel, so allow a
    somewhat looser tolerance than the seam test.
    """
    grid_res = 1.0
    smooth_deg = 5.0
    hot_value = 1000.0
    cold_value = 0.0

    # Equatorial pair: 0.6 true degrees apart (no cos(dec) shrinkage at Dec=0).
    ra_eq = np.array([0.0, 0.6])
    dec_eq = np.array([0.0, 0.0])
    values_eq = np.array([hot_value, cold_value])

    # Polar pair: 36 raw RA degrees apart, ~0.63 true degrees apart at Dec=89.
    ra_polar = np.array([0.0, 36.0])
    dec_polar = np.array([89.0, 89.0])
    values_polar = np.array([hot_value, cold_value])

    lon_eq, dec_grid_eq, smoothed_eq = grid_and_smooth(
        ra_eq, dec_eq, values_eq, smooth_deg, grid_resolution=grid_res
    )
    lon_p, dec_grid_p, smoothed_p = grid_and_smooth(
        ra_polar, dec_polar, values_polar, smooth_deg, grid_resolution=grid_res
    )

    peak_eq = _smoothed_value_near(lon_eq, dec_grid_eq, smoothed_eq, 0.0, 0.0)
    peak_polar = _smoothed_value_near(lon_p, dec_grid_p, smoothed_p, 0.0, 89.0)

    # True angular separation is what should matter, so both pairs should
    # be smoothed to a similar degree (measured diff ~4.3 out of ~500 with
    # the cos(dec)-scaled kernel, vs. ~500 out of 1000 pre-fix).
    assert peak_polar == pytest.approx(peak_eq, abs=10.0), (
        f"Hot point peak at high declination ({peak_polar:.1f}) was smoothed "
        f"much less than the equatorial pair at the same true angular "
        f"separation ({peak_eq:.1f}); the projection isn't correcting for "
        f"cos(dec) shrinkage of RA, distorting dynamic range near the poles."
    )
