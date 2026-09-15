"""
NOTE: THIS UNIT TEST WAS WRITTEN BY CLAUDE.

Unit tests for grid_and_smooth() in rednoise_skymap.py, targeting artifacts
of its sky projection that distort dynamic range in the resulting skymap,
ahead of the Mollweide plot.

grid_and_smooth() evaluates, at every point of a regular lon/dec display
grid, a real-space Gaussian-weighted average ("kernel regression") of
`values` at every actual input pointing within `mask_radius_deg`, weighted
by true great-circle distance. Because this is a genuine weighted MEAN of
nearby data (not an accumulate-then-divide-by-approximately-the-same-
denominator ratio), it has three properties these tests pin down:

1. Density independence: if every pointing within reach of a display cell
   shares the same true value V, the weighted average returns exactly V
   regardless of how many pointings contribute -- one or a thousand.
   Coverage density only ever decides *which* cells have a value at all,
   never what that value *is* (test_density_does_not_bias_the_value). This
   was a real, measured bug in an earlier HEALPix-based implementation
   (bin -> hp.smoothing() -> hp.get_interp_val()): hp.smoothing()'s
   spherical-harmonic transform is only an approximation of a true
   continuous convolution (band-limited to a finite lmax), and that
   approximation didn't cancel exactly between the numerator and
   denominator of the old accumulate-then-divide-by-coverage ratio right
   at a sharp coverage-density edge -- producing values measurably biased
   toward zero at the tapering edge of a real, gappy survey footprint,
   even though the true value there should not depend on local density at
   all.

2. No RA=180 seam: true great-circle distance between two points has no
   notion of an RA=0/360 array edge for two nearby points to land on
   opposite sides of (test_ra_180_seam_preserves_contrast_that_should_be_smoothed_out).

3. No polar distortion: true great-circle distance is not a fixed-degree-
   of-RA kernel, so it doesn't cover less true sky near the poles than at
   the equator the way a naive RA/Dec Cartesian grid would
   (test_high_declination_kernel_covers_larger_true_sky_area).

Properties 2 and 3 were originally written (and verified to fail loudly,
peak differing by ~500 out of 1000, i.e. no smoothing at all across the
seam/pole) against a pre-HEALPix, plain Cartesian-grid implementation, and
still hold for this real-space-distance implementation for the same
underlying reason HEALPix was originally introduced to fix them: a
correctly computed angular distance is inherently free of both artifacts,
independent of *how* that distance then gets used.

This file is self-contained: it locates and imports rednoise_skymap.py by
walking up from its own location on disk (not the current working
directory), so `pytest test_rednoise_skymap.py -v` works no matter which
directory you run it from, and even if this file gets copied/moved
somewhere else inside the repo. It also stubs out h5py/sps_common if either
isn't installed, since rednoise_skymap.py imports both at module scope
(for load_exposure_lookup()/TSAMP) even though grid_and_smooth(), the
function under test here, never touches them. Everything grid_and_smooth()
itself actually uses (numpy, scipy.spatial.cKDTree) is a real,
non-optional dependency of the module, so none of it is stubbed or skipped
here -- these tests exercise the real implementation directly.
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
    Control/sanity check: a spatially uniform field, evaluated somewhere
    with no seam and no pole nearby, should come back out exactly uniform
    -- a real-space weighted average of a constant field is that constant,
    exactly, with no approximation involved. This isolates the artifacts
    below from ordinary, expected smoothing/coverage-normalization
    behaviour.
    """
    rng = np.random.default_rng(0)
    n = 400
    ra = rng.uniform(60.0, 120.0, n)    # well clear of the RA=180 seam
    dec = rng.uniform(-20.0, 20.0, n)   # well clear of the poles
    value = np.full(n, 42.0)

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, value, smooth_deg=3.0, display_resolution=0.5
    )

    finite = smoothed[np.isfinite(smoothed)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 42.0, atol=1e-9)


def test_density_does_not_bias_the_value():
    """
    The core correctness property of a real-space kernel-weighted average:
    coverage density should only ever decide *where* we have a displayed
    value, never *what* that value is. A display cell right at the edge of
    a sparsely-covered region, backed by only one or two real pointings,
    must show the same true value as a cell deep inside a densely-covered
    region of identical true intensity -- not something biased toward zero
    just because fewer pointings happen to be nearby.

    This is precisely the property an earlier HEALPix-based implementation
    got measurably wrong: hp.smoothing()'s band-limited spherical-harmonic
    transform doesn't cancel exactly between the accumulated-value map and
    the coverage-count map at a sharp density edge, biasing the ratio
    toward zero right where density tapers off. A real-space weighted mean
    has no such bias by construction, which this test confirms directly by
    comparing a sparse region (2 contributing pointings) against a dense
    region (400 contributing pointings) of the same true value.
    """
    smooth_deg = 2.0
    true_value = 17.0

    # Dense region: many pointings, all the same true value, clustered
    # tightly enough that a display cell at the center sees strong local
    # density.
    rng = np.random.default_rng(1)
    ra_dense = 100.0 + rng.uniform(-0.3, 0.3, 400)
    dec_dense = 0.0 + rng.uniform(-0.3, 0.3, 400)
    values_dense = np.full(400, true_value)

    # Sparse region: only 2 pointings, same true value, same local
    # geometry otherwise (far from any seam/pole).
    ra_sparse = np.array([200.0, 200.2])
    dec_sparse = np.array([0.0, 0.0])
    values_sparse = np.full(2, true_value)

    ra = np.concatenate([ra_dense, ra_sparse])
    dec = np.concatenate([dec_dense, dec_sparse])
    values = np.concatenate([values_dense, values_sparse])

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=smooth_deg, display_resolution=0.25,
        mask_radius_deg=4.0,
    )

    dense_val = _smoothed_value_near(lon_grid, dec_grid, smoothed, 100.0, 0.0)
    sparse_val = _smoothed_value_near(lon_grid, dec_grid, smoothed, 200.1, 0.0)

    assert np.isfinite(dense_val) and np.isfinite(sparse_val)
    assert dense_val == pytest.approx(true_value, abs=1e-6), (
        f"Dense-region value ({dense_val}) should exactly reproduce the "
        f"uniform true value ({true_value})."
    )
    assert sparse_val == pytest.approx(true_value, abs=1e-6), (
        f"Sparse-region value ({sparse_val}) should *also* exactly "
        f"reproduce the uniform true value ({true_value}) -- coverage "
        f"density (400 nearby pointings vs. 2) should not bias the "
        f"displayed value at all, only whether a cell has one."
    )


def test_ra_180_seam_preserves_contrast_that_should_be_smoothed_out():
    """
    A hot point and a cold point 2 degrees apart should be blended by
    Gaussian smoothing (5 deg FWHM) by roughly the same amount no matter
    *where* on the sky that pair sits. Here the identical pair is placed
    once away from any wrap boundary (control) and once straddling RA=180
    (seam). True great-circle distance has no notion of an RA=0/360 array
    edge, so the seam pair should be blended together just as well as the
    control pair, since 179 deg and 181 deg are only 2 degrees apart on
    the sky, same as 89 deg and 91 deg.
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

    lon_c, dec_c, smoothed_c = grid_and_smooth(ra_control, dec_control, values_control, smooth_deg)
    lon_s, dec_s, smoothed_s = grid_and_smooth(ra_seam, dec_seam, values_seam, smooth_deg)

    peak_control = _smoothed_value_near(lon_c, dec_c, smoothed_c, 89.0, 0.0)
    peak_seam = _smoothed_value_near(lon_s, dec_s, smoothed_s, 179.0, 0.0)

    # The two configurations are related by a pure rotation in longitude,
    # and a true great-circle distance calculation has no privileged
    # longitude, so these should now match essentially exactly (up to
    # floating point and the query point snapping to the nearest display
    # grid cell) -- not just "roughly," the way the earlier band-limited
    # HEALPix implementation only approximately achieved.
    assert peak_seam == pytest.approx(peak_control, abs=0.5), (
        f"Hot point peak near the RA=180 seam ({peak_seam:.1f}) retained "
        f"more of its raw contrast than the seam-free control "
        f"({peak_control:.1f}); the distance calculation isn't treating "
        f"RA as periodic, artificially inflating dynamic range at the seam."
    )


def test_high_declination_kernel_covers_larger_true_sky_area():
    """
    Two point sources at the same *true* angular separation should be
    smoothed together to a similar degree regardless of declination. The
    polar pair here is 36 raw RA degrees apart, but at Dec=89 that's
    36 * cos(89 deg) =~ 0.63 true degrees -- essentially the same true
    separation as the equatorial pair, placed 0.6 degrees apart in raw RA.
    A true great-circle distance calculation doesn't shrink RA near the
    poles the way a naive Cartesian RA/Dec grid does, so both pairs should
    now be blended by essentially the same amount -- unlike a
    fixed-degree-of-RA kernel, which would barely touch the polar pair
    (36 raw degrees is far outside a 5 deg kernel) while heavily blending
    the equatorial one (0.6 deg is well inside it), a ~500-out-of-1000
    discrepancy.
    """
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

    lon_eq, dec_grid_eq, smoothed_eq = grid_and_smooth(ra_eq, dec_eq, values_eq, smooth_deg)
    lon_p, dec_grid_p, smoothed_p = grid_and_smooth(ra_polar, dec_polar, values_polar, smooth_deg)

    peak_eq = _smoothed_value_near(lon_eq, dec_grid_eq, smoothed_eq, 0.0, 0.0)
    peak_polar = _smoothed_value_near(lon_p, dec_grid_p, smoothed_p, 0.0, 89.0)

    # True angular separation is what should matter, so both pairs should
    # now be smoothed to essentially the same degree.
    assert peak_polar == pytest.approx(peak_eq, abs=5.0), (
        f"Hot point peak at high declination ({peak_polar:.1f}) was smoothed "
        f"much less than the equatorial pair at the same true angular "
        f"separation ({peak_eq:.1f}); the distance calculation isn't "
        f"correcting for cos(dec) shrinkage of RA, distorting dynamic "
        f"range near the poles."
    )


def test_sparse_coverage_does_not_blank_out_display_grid_near_data():
    """
    Regression test for a real bug found running against a full-scale
    survey file: a small, densely-covered patch of sky (as real data looks
    like almost everywhere on a partial-sky, drift-scan survey footprint)
    used to come back as all-NaN, including at the display cell sitting
    right on top of the real data. That was specific to the earlier
    HEALPix-based implementation's design (masking to NaN in HEALPix space
    before interpolating, which poisoned hp.get_interp_val()'s neighbor
    blending almost everywhere on a mostly-uncovered sky) -- this real-
    space implementation has no interpolation step to poison in the first
    place, but the test is kept as a straightforward correctness check on
    the replacement.
    """
    # A tiny, compact cluster of real coverage -- an "island" surrounded by
    # a totally empty sky, standing in for a small patch of real pointings.
    ra = np.array([10.0, 10.3, 9.7, 10.0])
    dec = np.array([20.0, 20.0, 20.0, 20.3])
    values = np.array([5.0, 5.0, 5.0, 5.0])

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=1.0, display_resolution=1.0
    )

    peak = _smoothed_value_near(lon_grid, dec_grid, smoothed, 10.0, 20.0)
    assert np.isfinite(peak), (
        f"grid_and_smooth() returned {peak} (not finite) at a display cell "
        f"sitting right on top of real, well-covered data."
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


def test_single_non_finite_value_does_not_poison_a_wide_area():
    """
    Regression test for a second real bug, found on a full-scale survey
    file after the sparse-coverage fix above: the whole map still came
    back all-NaN. Root cause at the time: hp.smoothing() computes a
    spherical-harmonic transform with *global* support (every output pixel
    a function of every input pixel), so a single non-finite value
    anywhere in `values` (e.g. one pointing-day out of 1.5 million whose
    upstream median_across_dms row was NaN because it had zero valid DMs
    recorded) poisoned the transform for the *entire sky*.

    This real-space implementation has no global transform -- a bad value
    can now only poison display cells within mask_radius_deg of it, via
    the weighted sum it contributes to -- but grid_and_smooth() still
    drops non-finite rows defensively up front (load_data() also does
    this earlier, for non-finite rn_sum specifically, so this is
    defense-in-depth on grid_and_smooth() itself, not reliant on every
    caller having already cleaned its inputs). This test checks both that
    the guard fires (a region far from the bad point is completely
    unaffected) and that it doesn't over-trigger (the rest of the map is
    still populated).
    """
    rng = np.random.default_rng(3)
    n = 40
    ra = rng.uniform(0.0, 60.0, n)
    dec = rng.uniform(-20.0, 20.0, n)
    values = rng.uniform(1.0, 10.0, n)
    values[7] = np.nan  # one bad pointing-day, same as a zero-valid-DM row

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=3.0, display_resolution=1.0
    )

    n_finite = np.count_nonzero(np.isfinite(smoothed))
    assert n_finite > 0, (
        "A single non-finite value among 40 otherwise-good points wiped "
        "out every cell of the display grid. grid_and_smooth() should "
        "drop non-finite ra/dec/value rows before computing the weighted "
        "average rather than passing them through."
    )


def test_gap_between_coverage_islands_stays_blank():
    """
    A display cell in the middle of a wide, genuine gap between two small
    coverage islands must be masked out (NaN), not filled in with some
    diluted average of far-away data -- this is exactly what
    mask_radius_deg exists to guarantee, and (with a real-space distance
    computation, no harmonic transform involved) there is no ringing
    mechanism that could ever paint a spurious value there the way an
    earlier HEALPix-based implementation's coverage map could.
    """
    # Two small, well-separated islands of real coverage with a wide,
    # genuinely empty gap between them -- like real gaps between
    # drift-scan pointings.
    ra = np.array([0.0, 0.3, -0.3, 0.0, 30.0, 30.3, 29.7, 30.0])
    dec = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3])
    values = np.full(8, 5.0)

    lon_grid, dec_grid, smoothed = grid_and_smooth(
        ra, dec, values, smooth_deg=1.0, display_resolution=1.0,
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
        f"be masked out; got {mid_gap} instead."
    )
