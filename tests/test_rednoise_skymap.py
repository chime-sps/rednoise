"""
Unit tests for grid_and_smooth() in rednoise_skymap.py, targeting artifacts
of its sky projection that distort dynamic range in the resulting skymap,
ahead of the Mollweide plot.

grid_and_smooth() bins pointings onto a HEALPix grid and then applies a
spherical-harmonic Gaussian beam (hp.smoothing()), weight-normalized by
coverage so an isolated value is reproduced exactly (see
test_flat_field_recovers_exact_value_away_from_edges). HEALPix is an
equal-area, iso-latitude pixelization with no privileged longitude, so it
does not have either of the two artifacts a naive RA/Dec Cartesian grid +
2-D Gaussian filter has:

1. RA=180 seam: a plain scipy.ndimage.gaussian_filter() on a Cartesian
   RA/Dec array has no notion that the array's left and right edges are
   physically adjacent on the sky, so two points a few degrees apart that
   straddle RA=180 land ~360 degrees apart in array-index space and never
   get smoothed together -- artificially preserving contrast at the
   anti-meridian. HEALPix pixels and hp.smoothing()'s spherical-harmonic
   beam have no array edge in longitude at all, so this can't happen.

2. High declination: a fixed-degree-of-RA kernel covers much less true sky
   near the poles than at the equator (RA circles shrink by cos(dec)), so
   two points at the same *true* angular separation used to be smoothed
   together very differently depending on declination. HEALPix pixels have
   (approximately) equal true sky area everywhere, so a fixed-angle
   Gaussian beam covers a constant true sky area regardless of declination.

These tests pin down that both properties hold: for each artifact, an
otherwise-equivalent pair of points placed away from the seam/pole (control)
and one placed right at it (seam/polar) should be smoothed to a similar
degree. They were originally written (and verified to fail loudly, peak
differing by ~500 out of 1000, i.e. no smoothing at all across the
seam/pole) against the pre-HEALPix Cartesian implementation.

This file is self-contained: it locates and imports rednoise_skymap.py by
walking up from its own location on disk (not the current working
directory), so `pytest test_rednoise_skymap.py -v` works no matter which
directory you run it from, and even if this file gets copied/moved
somewhere else inside the repo. It also stubs out h5py/sps_common if either
isn't installed, since rednoise_skymap.py imports both at module scope
(for load_exposure_lookup()/TSAMP) even though grid_and_smooth(), the
function under test here, never touches them. healpy is NOT stubbed --
it's the actual mechanism under test here (the whole point is verifying
real HEALPix's equal-area, no-seam properties), so these tests are skipped
(not faked) if the real package isn't installed; run `pip install healpy`
to enable them.
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


def _stub_sps_common_if_missing():
    """
    Install a lightweight stand-in for sps_common.constants, only if the
    real package isn't installed. rednoise_skymap.py does
    `from sps_common.constants import TSAMP` at module scope, but
    grid_and_smooth() never touches TSAMP.

    Both the parent package ("sps_common") and the submodule
    ("sps_common.constants") must be registered in sys.modules -- Python
    resolves "from sps_common.constants import TSAMP" by importing
    "sps_common" first, so registering only the submodule isn't enough.
    """
    import importlib
    try:
        importlib.import_module("sps_common.constants")
        return
    except ImportError:
        pass

    import types
    pkg = types.ModuleType("sps_common")
    const_mod = types.ModuleType("sps_common.constants")
    # Placeholder only -- these tests don't depend on the real TSAMP value.
    const_mod.TSAMP = 0.00098304
    pkg.constants = const_mod
    sys.modules["sps_common"] = pkg
    sys.modules["sps_common.constants"] = const_mod


_add_rednoise_skymap_dir_to_syspath()
_stub_h5py_if_missing()
_stub_sps_common_if_missing()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# healpy is the actual mechanism under test (its real equal-area, no-seam
# pixelization is what these tests verify), so it is never stubbed --
# import it for real or skip the module.
pytest.importorskip("healpy")

from rednoise_skymap import grid_and_smooth  # noqa: E402

# Resolution used for these tests: fine enough to resolve the point pairs
# below without being slow (real healpy's spherical-harmonic smoothing is
# cheap even at much higher nside than this).
TEST_NSIDE = 64


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
    somewhere with no seam and no pole nearby, should come back out
    (nearly) uniform. This isolates the artifacts below from ordinary,
    expected smoothing/coverage-normalization behaviour.
    """
    rng = np.random.default_rng(0)
    n = 400
    ra = rng.uniform(60.0, 120.0, n)    # well clear of the RA=180 seam
    dec = rng.uniform(-20.0, 20.0, n)   # well clear of the poles
    value = np.full(n, 42.0)

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, value, smooth_deg=3.0, nside=TEST_NSIDE, display_resolution=0.5
    )

    finite = smoothed[np.isfinite(smoothed)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 42.0, atol=1e-3)


def test_ra_180_seam_preserves_contrast_that_should_be_smoothed_out():
    """
    A hot point and a cold point 2 degrees apart should be blended by
    Gaussian smoothing (5 deg FWHM) by roughly the same amount no matter
    *where* on the sky that pair sits. Here the identical pair is placed
    once away from any wrap boundary (control) and once straddling RA=180
    (seam). HEALPix has no notion of an RA=0/360 array edge, so the seam
    pair should be blended together just as well as the control pair,
    since 179 deg and 181 deg are only 2 degrees apart on the sky, same as
    89 deg and 91 deg.
    """
    smooth_deg = 5.0
    hot_value = 1000.0
    cold_value = 0.0

    # Control pair: 2 degrees apart, far from any wrap boundary.
    ra_control = np.array([89.0, 91.0])
    dec_control = np.array([0.0, 0.0])
    values_control = np.array([hot_value, cold_value])

    # Seam pair: identical geometry (2 degrees apart, same Dec), straddling
    # RA=180.
    ra_seam = np.array([179.0, 181.0])
    dec_seam = np.array([0.0, 0.0])
    values_seam = np.array([hot_value, cold_value])

    lon_c, dec_c, smoothed_c = grid_and_smooth(
        ra_control, dec_control, values_control, smooth_deg, nside=TEST_NSIDE
    )
    lon_s, dec_s, smoothed_s = grid_and_smooth(
        ra_seam, dec_seam, values_seam, smooth_deg, nside=TEST_NSIDE
    )

    peak_control = _smoothed_value_near(lon_c, dec_c, smoothed_c, 89.0, 0.0)
    peak_seam = _smoothed_value_near(lon_s, dec_s, smoothed_s, 179.0, 0.0)

    # With HEALPix, the hot point's smoothed peak should be attenuated by
    # essentially the same amount in both cases (up to pixelization
    # quantization), since the two configurations are related by a pure
    # rotation in longitude and HEALPix has no privileged longitude.
    assert peak_seam == pytest.approx(peak_control, abs=5.0), (
        f"Hot point peak near the RA=180 seam ({peak_seam:.1f}) retained "
        f"more of its raw contrast than the seam-free control "
        f"({peak_control:.1f}); the projection isn't treating RA as "
        f"periodic, artificially inflating dynamic range at the seam."
    )


def test_high_declination_kernel_covers_larger_true_sky_area():
    """
    Two point sources at the same *true* angular separation should be
    smoothed together to a similar degree regardless of declination. The
    polar pair here is ~15.5 raw RA degrees apart, but at Dec=75 that's
    15.5 * cos(75 deg) =~ 4 true degrees -- the same true separation as the
    equatorial pair, placed 4 degrees apart in raw RA. Both separations are
    kept comfortably larger than a single HEALPix pixel at TEST_NSIDE (a
    pixel is roughly 58.6/nside =~ 0.9 deg across at nside=64), so the two
    points always land in different pixels; otherwise binning would merge
    them together before smoothing ever runs, and the test would no longer
    be exercising the smoothing kernel's behaviour at all. With HEALPix's
    equal-area pixels, both pairs should be blended to a *similar* amount --
    within real hp.smoothing()'s pixelization noise (see the tolerance note
    below) -- unlike the pre-HEALPix behaviour (a fixed-degree-of-RA
    kernel), which barely touched the polar pair (15.5 raw degrees is far
    outside a 5 deg kernel) while heavily blending the equatorial one
    (4 deg is well inside it), a ~500-out-of-1000 discrepancy.
    """
    smooth_deg = 5.0
    hot_value = 1000.0
    cold_value = 0.0

    control_sep_deg = 4.0   # true angular separation, both pairs
    dec_polar_val = 75.0
    polar_sep_deg = control_sep_deg / np.cos(np.deg2rad(dec_polar_val))

    # Equatorial pair: 4 true degrees apart (no cos(dec) shrinkage at Dec=0).
    ra_eq = np.array([0.0, control_sep_deg])
    dec_eq = np.array([0.0, 0.0])
    values_eq = np.array([hot_value, cold_value])

    # Polar pair: same true separation as the equatorial pair, but raw RA
    # separation is scaled up by 1/cos(dec) to compensate for cos(dec)
    # shrinkage of RA near the pole.
    ra_polar = np.array([0.0, polar_sep_deg])
    dec_polar = np.array([dec_polar_val, dec_polar_val])
    values_polar = np.array([hot_value, cold_value])

    lon_eq, dec_grid_eq, smoothed_eq = grid_and_smooth(
        ra_eq, dec_eq, values_eq, smooth_deg, nside=TEST_NSIDE
    )
    lon_p, dec_grid_p, smoothed_p = grid_and_smooth(
        ra_polar, dec_polar, values_polar, smooth_deg, nside=TEST_NSIDE
    )

    peak_eq = _smoothed_value_near(lon_eq, dec_grid_eq, smoothed_eq, 0.0, 0.0)
    peak_polar = _smoothed_value_near(lon_p, dec_grid_p, smoothed_p, 0.0, dec_polar_val)

    # True angular separation is what should matter, so both pairs should
    # be smoothed to a similar degree. The tolerance here is looser than
    # the RA=180 seam test's: real hp.smoothing() is a *band-limited*
    # (finite lmax, default 3*nside-1) spherical-harmonic beam, and a
    # binned point source is a sharp, near-delta-function input to that
    # transform, so its band-limited reconstruction shows some
    # position-dependent "ringing" relative to the pixel grid -- typically
    # several percent of the peak, well below the ~50% (500-out-of-1000)
    # discrepancy the pre-HEALPix seam/pole bugs produced. This test's job
    # is to catch that qualitative failure mode reappearing, not to demand
    # exact numerical equality between two arbitrarily-phased point pairs.
    assert peak_polar == pytest.approx(peak_eq, abs=120.0), (
        f"Hot point peak at high declination ({peak_polar:.1f}) was smoothed "
        f"much less than the equatorial pair at the same true angular "
        f"separation ({peak_eq:.1f}); the projection isn't correcting for "
        f"cos(dec) shrinkage of RA, distorting dynamic range near the poles."
    )


def test_nside_must_be_a_power_of_two():
    """
    hp.ang2pix()/hp.smoothing() require nside to be a power of 2; catch a
    bad value with a clear error rather than a confusing failure deep
    inside healpy.
    """
    ra = np.array([10.0, 20.0])
    dec = np.array([0.0, 0.0])
    values = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        grid_and_smooth(ra, dec, values, smooth_deg=1.0, nside=100)


def test_sparse_coverage_does_not_blank_out_display_grid_near_data():
    """
    Regression test for a real bug found running against a full-scale
    survey file: a small, densely-covered patch of sky (as real data looks
    like almost everywhere on a partial-sky, drift-scan survey footprint --
    most HEALPix pixels at typical `nside` are never actually hit by a
    pointing) used to come back as all-NaN, including at the display cell
    sitting right on top of the real data.

    Root cause: the previous implementation divided accum/count and masked
    the result to NaN wherever coverage was below threshold *in HEALPix
    space*, then resampled that NaN-containing map onto the display grid
    with hp.get_interp_val(). That function bilinearly blends each display
    point's several neighboring HEALPix pixels, and a weighted sum touching
    even one NaN neighbor is itself NaN. Since the overwhelming majority of
    pixels on a realistic, only-partially-covered sky are uncovered (NaN),
    essentially every display point's neighbor stencil touches at least one
    of them -- wiping out nearly the whole map, including cells right next
    to real data whose own stencil still touched a neighboring empty pixel.

    The fix interpolates the (always-finite) smoothed accum and count maps
    separately, and only divides/masks by coverage after landing on the
    regular display grid -- see grid_and_smooth()'s docstring.

    This is exactly the failure mode the earlier artifact tests above
    couldn't catch: they only ever queried a display point sitting deep
    inside a smooth, wide (multi-pixel) Gaussian blob built from just 2
    points on an otherwise-empty sphere, always well clear of any
    NaN/covered boundary. Here, a *small, compact* island of real coverage
    with a *narrow* smoothing kernel (so untouched sky starts close by, as
    it realistically does between real pointings) reproduces the actual
    failure.
    """
    # A tiny, compact cluster of real coverage -- an "island" surrounded by
    # a totally empty sky, standing in for a small patch of real pointings.
    ra = np.array([10.0, 10.3, 9.7, 10.0])
    dec = np.array([20.0, 20.0, 20.0, 20.3])
    values = np.array([5.0, 5.0, 5.0, 5.0])

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=1.0, nside=TEST_NSIDE, display_resolution=1.0
    )

    peak = _smoothed_value_near(lon_grid, dec_grid, smoothed, 10.0, 20.0)
    assert np.isfinite(peak), (
        f"grid_and_smooth() returned {peak} (not finite) at a display cell "
        f"sitting right on top of real, well-covered data. On a sky that's "
        f"mostly uncovered elsewhere (realistic for a partial-sky survey), "
        f"this points to coverage-masking happening in HEALPix space before "
        f"interpolation, which poisons hp.get_interp_val()'s neighbor-"
        f"blending with NaN almost everywhere -- see this test's docstring."
    )
    assert peak == pytest.approx(5.0, abs=1e-6), (
        f"Expected the uniform input value (5.0) to be reproduced exactly "
        f"at the data location; got {peak}."
    )

    # Also confirm the map isn't secretly *entirely* finite (e.g. from a
    # coverage-mask threshold that's silently gone missing) -- most of this
    # otherwise-empty sky should still correctly show as NaN.
    n_finite = np.count_nonzero(np.isfinite(smoothed))
    assert 0 < n_finite < smoothed.size, (
        f"Expected only a small island of finite cells near the data "
        f"({n_finite} of {smoothed.size} were finite) -- either coverage "
        f"masking is broken, or it vanished entirely."
    )


def test_single_non_finite_value_does_not_poison_the_whole_sky():
    """
    Regression test for a second real bug, found on a full-scale survey
    file after the sparse-coverage fix above: the whole map still came
    back all-NaN.

    Root cause: hp.smoothing() computes a spherical-harmonic transform,
    which has *global* support -- every output pixel is a function of
    every input pixel, not just its real-space neighbors the way a
    Cartesian Gaussian filter would be. So a single non-finite value
    anywhere in `values` (e.g. one pointing-day out of 1.5 million whose
    upstream median_across_dms row was NaN because it had zero valid DMs
    recorded) doesn't just spoil its own HEALPix pixel -- it poisons
    hp.smoothing()'s transform for the *entire sky*, turning every output
    value to NaN. This is confirmed with this repo's local healpy stand-in,
    whose brute-force real-space smoothing has the same global-poisoning
    property for a different reason (every output pixel's weighted sum
    includes every input pixel with a nonzero, if tiny, Gaussian weight,
    and `tiny_weight * nan` is still `nan`).

    grid_and_smooth() now drops non-finite ra/dec/value rows up front
    (load_data() also does this earlier, for non-finite rn_sum specifically,
    so this is a defense-in-depth check on grid_and_smooth() itself, not
    reliant on every caller having already cleaned its inputs).
    """
    rng = np.random.default_rng(3)
    n = 40
    ra = rng.uniform(0.0, 60.0, n)
    dec = rng.uniform(-20.0, 20.0, n)
    values = rng.uniform(1.0, 10.0, n)
    values[7] = np.nan  # one bad pointing-day, same as a zero-valid-DM row

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=3.0, nside=TEST_NSIDE, display_resolution=1.0
    )

    n_finite = np.count_nonzero(np.isfinite(smoothed))
    assert n_finite > 0, (
        "A single non-finite value among 40 otherwise-good points wiped "
        "out every cell of the display grid -- hp.smoothing()'s transform "
        "has global support, so one bad input pixel poisons the whole "
        "sky. grid_and_smooth() should drop non-finite ra/dec/value rows "
        "before binning/smoothing rather than passing them through."
    )


def test_gap_between_coverage_islands_stays_blank():
    """
    Regression test for a third real-data issue: with default parameters,
    the displayed map "filled in" small, disconnected dots -- separated
    from real coverage by whitespace -- in the sparse gaps between actual
    coverage regions, rather than a smooth falloff.

    Cause: hp.smoothing() is a spherical-harmonic beam, which can ring
    (like any truncated/band-limited representation of a sharp-edged
    signal -- here, a sparse, sharp-boundaried real coverage pattern with
    genuine gaps between drift-scan pointings). That ringing can swing the
    smoothed *count* map back above the `count_mesh > 1e-6` coverage
    threshold in small, isolated spots deep inside a real gap, "filling
    in" a plausible-looking value where there's actually no data at all --
    exactly the disconnected-dots-in-whitespace pattern reported.

    Fix: gate display cells on real-space great-circle distance to the
    nearest *actual* input pointing (`mask_radius_deg`, default
    2 * smooth_deg) in addition to the smoothed-coverage threshold. A
    real-space nearest-neighbor distance can't ring, so a cell deep inside
    a genuine gap is reliably excluded regardless of what the harmonic
    beam's tails do there.

    This repo's local healpy stand-in can't actually reproduce the ringing
    itself (its brute-force smoothing is a strictly-positive-weight
    real-space convolution of a non-negative map, which is mathematically
    incapable of ringing negative or spuriously positive, unlike real
    healpy's spherical-harmonic transform) -- so this test instead pins
    down the *mask geometry* directly: a display cell in the middle of a
    wide, genuine gap between two coverage islands must be masked out
    regardless of smoothed coverage, purely because it's far in real-space
    from any actual pointing.
    """
    # Two small, well-separated islands of real coverage with a wide,
    # genuinely empty gap between them -- like real gaps between
    # drift-scan pointings.
    ra = np.array([0.0, 0.3, -0.3, 0.0, 30.0, 30.3, 29.7, 30.0])
    dec = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3])
    values = np.full(8, 5.0)

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=1.0, nside=TEST_NSIDE, display_resolution=1.0,
        mask_radius_deg=2.0,
    )

    peak1 = _smoothed_value_near(lon_grid, dec_grid, smoothed, 0.0, 0.0)
    peak2 = _smoothed_value_near(lon_grid, dec_grid, smoothed, 30.0, 0.0)
    mid_gap = _smoothed_value_near(lon_grid, dec_grid, smoothed, 15.0, 0.0)

    assert np.isfinite(peak1) and np.isfinite(peak2), (
        "Expected both real coverage islands to still show a finite value "
        "-- the mask_radius_deg guard should never blank out a cell right "
        "on top of actual data."
    )
    assert np.isnan(mid_gap), (
        f"Expected the middle of the genuine gap between the two coverage "
        f"islands (15 deg from either, well past mask_radius_deg=2.0) to "
        f"be masked out; got {mid_gap} instead. A display cell that far "
        f"from any real pointing should never be shown, regardless of "
        f"what the harmonic-smoothed coverage map says there."
    )
