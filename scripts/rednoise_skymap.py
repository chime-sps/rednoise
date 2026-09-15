#!/usr/bin/env python3
'''
This module reads a rednoise info file and produces the following plots:

    1. Skymap: mean across all days of the sum across all frequency bins > 5 of the
       median of the medians across all DMs. Each displayed sky position is a
       real-space, great-circle-distance Gaussian-weighted average ("kernel
       regression") of nearby pointings, with a customizable kernel width,
       then projected via Mollweide.
    2. Coverage map: one dot per saved pointing.

The mean across days is weighted by each day's exposure length T_exp, since
T_exp differs from pointing to pointing (it depends on the zenith angle at
transit). T_exp is derived from the "scales" array stored per pointing-day in
the combined medians HDF5 file produced by combine_rednoise_medians.py:

    N_in_power_spectrum = np.sum(scales)
    T_exp = 2 * N_in_power_spectrum * TSAMP

"scales" gives, for each (already rebinned) output frequency bin, how many
bins of the original power spectrum were averaged together to make it, so its
sum recovers the total number of bins in the original power spectrum. TSAMP
is the raw time-domain sampling interval, imported from sps_common.

Usage:

    python3 rednoise_skymap.py rednoise_dm_info.npz combined_medians.h5
    python3 rednoise_skymap.py rednoise_dm_info.npz combined_medians.h5 --smooth-deg 1.0
        --output-skymap skymap.png --output-coverage coverage.png
'''

from pathlib import Path

import click
import h5py
import numpy as np
from scipy.spatial import cKDTree

# my obsessive matplotlib formatting:
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'monospace'
from matplotlib.colors import Normalize, LogNorm

from sps_common.constants import TSAMP


# -------------
# load our data!
# -------------

def load_exposure_lookup(h5_path: Path):
    """
    Build a lookup table of exposure length T_exp (s) for every pointing-day
    stored in the combined medians HDF5 file.

    The combined medians file (see combine_rednoise_medians.py) stores one
    row per pointing-day, with a "scale" (sometimes "scales") array of shape
    (N, max_n_freq) giving the number of original power-spectrum bins that
    were rebinned into each saved frequency bin, -1 padded past each row's
    valid n_freq entries.

    Inputs:
    -------
        h5_path (Path): path to the combined medians HDF5 file

    Returns:
    --------
        lookup (dict): maps (year, month, day, RA rounded to 4 dp,
                        Dec rounded to 4 dp) -> T_exp (float, seconds)
    """
    print(f"Loading exposure info from {h5_path} ...", flush=True)

    with h5py.File(h5_path, "r") as h5:
        scale_key = "scales" if "scales" in h5 else "scale"
        scale = np.asarray(h5[scale_key])   # shape: (N, max_n_freq), -1 padded
        n_freq = np.asarray(h5["n_freq"])   # shape: (N,)
        ra = np.asarray(h5["ra"])
        dec = np.asarray(h5["dec"])
        year = np.asarray(h5["year"]).astype(int)
        month = np.asarray(h5["month"]).astype(int)
        day = np.asarray(h5["day"]).astype(int)

    n_rows = len(ra)
    t_exp = np.zeros(n_rows, dtype=np.float64)

    for i in range(n_rows):
        valid_scale = scale[i, :n_freq[i]]
        valid_scale = valid_scale[valid_scale >= 0]  # guard against stray padding
        n_in_power_spectrum = np.sum(valid_scale)
        t_exp[i] = 2.0 * n_in_power_spectrum * TSAMP

    keys = zip(year, month, day, np.round(ra, 4), np.round(dec, 4))
    lookup = dict(zip(keys, t_exp))

    print(f"  Built exposure lookup for {len(lookup)} pointing-day(s).", flush=True)

    return lookup


def load_data(npz_path: Path, h5_path: Path):
    """
    This function loads the DM info file and returns the relevant contents,
    with the per-pointing rednoise mean weighted by each day's exposure
    length T_exp (derived from the combined medians HDF5 file).

    Columns: RA, Dec, year, month, day, 0 if wonky RFI in first bins / 1 if not, DM where wonky behavior starts

    Inputs:
    -------
        npz_path (Path): path to the DM info file
        h5_path (Path): path to the combined medians HDF5 file (used to
            recover each pointing-day's exposure length)

    Returns:
    --------
        RA (arr) : RA of each pointing
        Dec (arr) : Dec of each pointing
        mean_rn (arr): exposure-weighted mean across days of sum across freq bins of median across DMs
    """

    print(f"Loading {npz_path} ...", flush=True)
    data = np.load(npz_path)

    info = data["info"]                            # shape: (N, 7)
    median_across_dms = data["median_across_dms"]  # shape: (N, max_n_freq)

    ra = info[:, 0]
    dec = info[:, 1]
    year = info[:, 2].astype(int)
    month = info[:, 3].astype(int)
    day = info[:, 4].astype(int)

    # take sum of median of medians across frequency bins after 5
    rn_sum = np.sum(median_across_dms[:, 5:], axis=1)  # shape: (N)

    # A row whose DM info file entry recorded zero valid DMs for that
    # pointing-day (n_dm == 0, e.g. a short/corrupted observation) has
    # np.median() of an empty slice baked into its median_across_dms row,
    # which is NaN -- so rn_sum is NaN for that row too. This is rare but
    # real at survey scale (millions of rows). grid_and_smooth()'s kernel
    # regression would fold a NaN value into the weighted average of every
    # display cell within reach of that one pointing, NaN-ing out an entire
    # (otherwise perfectly good) neighborhood around it. Drop those rows
    # here, with the count surfaced, rather than letting them propagate
    # into a confusing failure several functions away.
    bad_rn = ~np.isfinite(rn_sum)
    n_bad_rn = np.sum(bad_rn)
    if n_bad_rn:
        print(
            f"  [WARN] {n_bad_rn}/{len(rn_sum)} row(s) had a non-finite "
            f"median_across_dms sum (likely a pointing-day with zero valid "
            f"DMs recorded upstream) and are being dropped.",
            flush=True,
        )
        keep = ~bad_rn
        ra, dec = ra[keep], dec[keep]
        year, month, day = year[keep], month[keep], day[keep]
        rn_sum = rn_sum[keep]

    # exposure length (in seconds) for each row, from the combined medians h5 file
    exposure_lookup = load_exposure_lookup(h5_path)

    row_keys = list(zip(year, month, day, np.round(ra, 4), np.round(dec, 4)))
    t_exp = np.array([exposure_lookup.get(k, np.nan) for k in row_keys], dtype=np.float64)

    n_missing = np.sum(np.isnan(t_exp))
    if n_missing:
        print(
            f"  [WARN] {n_missing}/{len(t_exp)} row(s) had no matching entry in "
            f"{h5_path}; falling back to an unweighted contribution for those rows.",
            flush=True,
        )
        # Fall back to equal weighting (1.0) for rows we couldn't match, rather
        # than dropping them, so a partial h5 file doesn't silently discard data.
        t_exp = np.where(np.isnan(t_exp), 1.0, t_exp)

    # figure out how many pointings we have
    # consider 4 decimal places to be the "same"
    pointing_keys = np.round(ra, 4) * 1000 + np.round(dec, 4)
    unique_keys, inv = np.unique(pointing_keys, return_inverse=True)
    n_pointings = len(unique_keys)

    mean_rn = np.zeros(n_pointings, dtype=np.float64)
    weight_sum = np.zeros(n_pointings, dtype=np.float64)
    count = np.zeros(n_pointings, dtype=np.int32)
    ra_pt = np.zeros(n_pointings, dtype=np.float64)   # basically the mean RA across near matches
    dec_pt = np.zeros(n_pointings, dtype=np.float64)  # same here

    # weight each day's contribution to its pointing's mean by that day's T_exp
    np.add.at(mean_rn, inv, rn_sum * t_exp)
    np.add.at(weight_sum, inv, t_exp)
    np.add.at(count, inv, 1)
    np.add.at(ra_pt, inv, ra)
    np.add.at(dec_pt, inv, dec)

    mean_rn /= np.maximum(weight_sum, np.finfo(np.float64).tiny)
    ra_pt /= np.maximum(count, 1)
    dec_pt /= np.maximum(count, 1)

    print(f" {len(info)} rows -> {n_pointings} unique pointings.", flush=True)
    print(f" RA range: {ra_pt.min():.2f} -- {ra_pt.max():.2f} deg", flush=True)
    print(f" Dec range: {dec_pt.min():.2f} -- {dec_pt.max():.2f} deg", flush=True)

    return ra_pt, dec_pt, mean_rn


# --------
# OK now we are doing our fun density stuff!
# --------

def grid_and_smooth(ra, dec, values, smooth_deg, display_resolution=0.25,
                     mask_radius_deg=None):
    '''
    This function evaluates a real-space Gaussian-weighted average ("kernel
    regression") of `values` at every point of a regular lon/dec display
    grid, using true great-circle distance from each grid point to every
    actual input pointing.

    Concretely, for a display cell at true angular distance d_i from input
    pointing i (i = 1..N), the displayed value is:

        sum_i( w_i * values_i ) / sum_i( w_i ),   w_i = exp(-0.5 * (d_i / sigma)^2)

    with sigma derived from `smooth_deg` (a FWHM, sigma = FWHM / sqrt(8 ln 2)),
    and the sum restricted to pointings within `mask_radius_deg` (points
    farther away contribute a weight of ~0 anyway; a display cell with NO
    pointing within `mask_radius_deg` is left as NaN rather than assigning
    it some vanishingly diluted, physically meaningless average of far-away
    data).

    This is a weighted MEAN of nearby data, which is why it is inherently
    density-independent: if every pointing within reach of a display cell
    happens to share the same true value V, the weighted mean returns
    exactly V regardless of whether there's one contributing pointing or a
    thousand -- coverage density only ever decides *which* cells have any
    contributing pointings at all (and therefore get a value instead of
    NaN), never what that value *is*. Real great-circle distance is also
    inherently free of the two projection artifacts a naive RA/Dec
    Cartesian grid has: there's no RA=0/360 array edge for an angular
    distance calculation to trip over, and no cos(dec) foreshortening of
    RA near the poles either.

    This replaces an earlier HEALPix-based implementation
    (bin -> hp.smoothing() -> hp.get_interp_val()), dropped for a real,
    structural problem beyond the two implementation bugs already fixed in
    it (NaN poisoning the whole sky; harmonic ringing "filling in" spurious
    disconnected dots in real gaps): hp.smoothing()'s spherical-harmonic
    transform is only an *approximation* of a true continuous convolution,
    band-limited to a finite lmax, and that approximation does not cancel
    exactly between the numerator (accumulated value) and denominator
    (coverage count) of the old accumulate-then-divide-by-coverage ratio
    right at a sharp coverage-density edge -- which is exactly what a real,
    gappy drift-scan survey footprint looks like. The result was values
    measurably biased toward zero at the tapering edge of a coverage
    region, even though the true, density-independent value there should
    equal whatever nearby pointings actually measured. An exact real-space
    weighted average has no such bias, by construction, and removes the
    healpy dependency entirely in the process.

    Inputs:
    -------
        ra (arr): 1D array of RAs
        dec (arr): 1D array of Decs
        values (arr): mean_rn from load_data()
        smooth_deg (float): Gaussian kernel FWHM, in degrees
        display_resolution (float): spacing, in degrees, of the regular
            lon/dec grid the kernel regression is evaluated on
        mask_radius_deg (float or None): a display cell is assigned a value
            only if at least one actual input pointing lies within this
            great-circle distance; farther cells are NaN. Defaults to
            2 * smooth_deg (two beam widths) if None. This doubles as the
            kernel's hard cutoff radius for tractability: at 2 * smooth_deg
            a Gaussian's weight is already down to ~1.5e-5 of its peak, so
            excluding farther pointings from the weighted average entirely
            has no visible effect on the result -- unlike hp.smoothing()'s
            harmonic-space truncation, a real-space hard cutoff of an
            already-negligible tail doesn't ring.

    Returns:
    --------
        lon_grid (arr): 1-D array of longitude bin centres in degrees, monotonically
            increasing from negative (east) to positive (west), spanning the
            full circle (RA is periodic).
        dec_grid (arr): 1-D array of Dec bin centres in degrees
        smoothed (arr): 2-D array (n_dec, n_lon) of the kernel-weighted
            average value at each display cell, NaN where no pointing is
            within mask_radius_deg
    '''
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Defensive guard, independent of whatever the caller already did: a
    # single non-finite ra/dec/value would fold a NaN into the weighted sum
    # of every display cell within mask_radius_deg of it, NaN-ing out that
    # whole neighborhood. Drop such rows here so this function is safe on
    # its own, whatever the caller's input hygiene looks like.
    bad = ~(np.isfinite(ra) & np.isfinite(dec) & np.isfinite(values))
    if np.any(bad):
        print(
            f"  [WARN] grid_and_smooth: dropping {np.sum(bad)}/{len(bad)} "
            f"row(s) with non-finite ra/dec/value.",
            flush=True,
        )
        ra, dec, values = ra[~bad], dec[~bad], values[~bad]

    if mask_radius_deg is None:
        mask_radius_deg = 2.0 * smooth_deg

    sigma_deg = smooth_deg / np.sqrt(8.0 * np.log(2.0))  # FWHM -> sigma

    # Colatitude (0 at the north pole, pi at the south pole) and longitude
    # in [0, 2*pi), both in radians -- then straight to 3D unit vectors, so
    # every distance below is an exact great-circle (chord) distance, with
    # no seam and no polar distortion.
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra % 360.0)
    data_xyz = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Build the regular lon/dec display grid, purely for compatibility with
    # the existing matplotlib mollweide + pcolormesh display code. Gridding
    # is done in longitude degrees (lon = -(ra % 360), wrapped to
    # (-180, 180]), not in raw RA degrees -- otherwise pcolormesh throws a
    # hissy fit -- and covers the full circle since longitude is periodic.
    n_lon = max(int(round(360.0 / display_resolution)), 1)
    lon_resolution = 360.0 / n_lon
    lon_grid = -180.0 + lon_resolution * np.arange(n_lon)

    # shouldn't be any weird data past +90 dec but just to be sure
    dec_min, dec_max = max(dec.min() - 1, -90.0), min(dec.max() + 1, 90.0)
    dec_grid = np.arange(dec_min, dec_max + display_resolution, display_resolution)

    lon_mesh, dec_mesh = np.meshgrid(lon_grid, dec_grid)  # each (n_dec, n_lon)
    ra_mesh = np.mod(-lon_mesh, 360.0)
    theta_mesh = np.deg2rad(90.0 - dec_mesh)
    phi_mesh = np.deg2rad(ra_mesh)
    mesh_xyz = np.column_stack([
        np.sin(theta_mesh.ravel()) * np.cos(phi_mesh.ravel()),
        np.sin(theta_mesh.ravel()) * np.sin(phi_mesh.ravel()),
        np.cos(theta_mesh.ravel()),
    ])

    # "Splat" each actual pointing's Gaussian-weighted contribution onto
    # the (usually few) nearby display cells within mask_radius_deg, rather
    # than the other way around (checking every display cell against every
    # pointing) -- there are normally far fewer real pointings than display
    # cells, so this is the cheaper direction to loop over.
    n_cells = mesh_xyz.shape[0]
    weighted_sum = np.zeros(n_cells, dtype=np.float64)
    weight_total = np.zeros(n_cells, dtype=np.float64)

    max_chord = 2.0 * np.sin(np.deg2rad(mask_radius_deg) / 2.0)
    sigma_rad = np.deg2rad(sigma_deg)

    grid_tree = cKDTree(mesh_xyz)
    # One batched query for every pointing's nearby-cell list, rather than
    # N separate Python-level calls.
    neighbor_lists = grid_tree.query_ball_point(data_xyz, r=max_chord)

    for i, cell_idx in enumerate(neighbor_lists):
        if not cell_idx:
            continue
        cell_idx = np.asarray(cell_idx)
        chord = np.linalg.norm(mesh_xyz[cell_idx] - data_xyz[i], axis=1)
        # chord length between two unit vectors separated by true angle a
        # is 2*sin(a/2); invert that to recover the real angle for the
        # Gaussian weight (chord distance alone would distort the kernel
        # width away from the poles/across long distances).
        angle_rad = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
        w = np.exp(-0.5 * (angle_rad / sigma_rad) ** 2)
        np.add.at(weighted_sum, cell_idx, w * values[i])
        np.add.at(weight_total, cell_idx, w)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed_flat = np.where(weight_total > 1e-12, weighted_sum / weight_total, np.nan)

    smoothed = smoothed_flat.reshape(theta_mesh.shape)

    return lon_grid, dec_grid, smoothed


# --------
# argh argh coordinate conversion!
# --------

def ra_deg_to_moll_rad(ra_deg):
    '''
    This function converts RA in degrees to Mollweide longitude in radians.
    We flip the sign of the longitude from RA because in astro we increase RA to the left...

    lon_rad = -deg2rad(ra_deg % 360)

    Inputs:
    ------
        ra_deg (float): RA in degrees

    Returns:
    --------
        lon_rad (float): longitude in radians
    '''
    lon_rad = -np.deg2rad(ra_deg % 360.0)
    lon_rad = (lon_rad + np.pi) % (2 * np.pi) - np.pi
    return lon_rad


def dec_deg_to_moll_rad(dec_deg):
    '''
    I don't think I need to explain this one.
    '''
    return np.deg2rad(dec_deg)


def ra_deg_to_hours(ra_deg):
    '''
    Same here.
    '''
    return ra_deg / 15.0


# -------
# plotting stuff
# -------

def _setup_axes(dpi=150):
    '''
    This function sets up the figure and the Mollweide axes!

    Inputs:
    -------
        dpi (float): figure resolution, deliberately used for the
            *interactive* plt.show() window too, not just plt.savefig().
            A pcolormesh grid this fine can render differently at
            different resolutions: a small, faint, real coverage island
            can get anti-aliased into invisibility in a lower-resolution
            interactive window while still resolving clearly in a
            higher-dpi saved file (or vice versa) -- same data, same
            code path, different pixel grid to rasterize onto. Passing the
            same dpi to both makes plt.show() and plt.savefig() agree.
    '''
    fig = plt.figure(figsize=(14, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor('white')
    ax.grid(True, linestyle=":", alpha=0.6, color="black")

    # my custom xtick marks:
    ax.set_xticklabels([])
    hour_ticks_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    hour_ticks_rad = np.deg2rad(hour_ticks_deg)
    ra_hour_labels = ((-hour_ticks_deg) % 360) / 15.0
    label_dec_rad = np.deg2rad(-15)  # can toggle as desired

    for lon_rad, label in zip(hour_ticks_rad, ra_hour_labels):
        ax.text(lon_rad, label_dec_rad, f"{label:.0f}h",
                ha="center", va="top", fontsize=14, color="black")

    ax.set_xlabel("Right Ascension", labelpad=14, fontsize=13)
    ax.set_ylabel("Declination (deg)", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)

    return fig, ax


def plot_skymap(ra, dec, mean_rn, smooth_deg, display_res=0.25,
                 mask_radius_deg=None, dpi=150, output_path=None):
    '''
    This function creates and then plots the rednoise skymap.

    Inputs:
    ------
        ra (arr)
        dec (arr)
        mean_rn (arr)
        smooth_deg (float): Gaussian kernel FWHM in degrees
        display_res (float): regular lon/dec display-grid spacing in degrees
        mask_radius_deg (float or None): see grid_and_smooth(); defaults to
            2 * smooth_deg if None
        dpi (float): figure resolution, used for *both* plt.show() and
            plt.savefig() -- see _setup_axes()'s docstring for why an
            interactive window and a saved file need to share one dpi
            value here, unlike a typical matplotlib script.
        output_path (str): optional path to save image if desired
    '''
    print("Gridding and smoothing ...", flush=True)
    lon_grid, dec_grid, smoothed = grid_and_smooth(ra, dec, mean_rn, smooth_deg,
                                                    display_resolution=display_res,
                                                    mask_radius_deg=mask_radius_deg)

    fig, ax = _setup_axes(dpi=dpi)

    finite = smoothed[np.isfinite(smoothed)]
    finite_pos = finite[finite > 0]

    # NOTE: a genuinely covered display cell whose smoothed value happens to
    # sit near/below this 2nd-percentile vmin (e.g. right at the tapering
    # edge of a sparse coverage island) is NOT masked out -- it's clipped to
    # the bottom of the colormap by LogNorm, which for "inferno" is
    # near-black. That's expected/correct (distinguishing real-but-faint
    # coverage, near-black, from genuinely no-data, white via
    # cmap.set_bad() below) -- but it's easy to mistake for a rendering bug
    # if it only becomes visible in a saved file and not on an interactive
    # plot rendered at a different (often lower) screen resolution; see the
    # `dpi` parameter above and _setup_axes()'s docstring.
    #if len(finite_pos) == 0:
    #    norm = Normalize(vmin=np.percentile(finite, 2),
    #                      vmax=np.percentile(finite, 98))
    #else:
    norm = LogNorm(vmin=np.percentile(finite_pos, 2),
                    vmax=np.percentile(finite_pos, 98))

    # convert to radians for pcolormesh... could just save as radians initially
    # maybe easier
    lon_rad = np.deg2rad(lon_grid)
    dec_rad = np.deg2rad(dec_grid)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad("white")  # could add toggle

    pcm = ax.pcolormesh(lon_rad, dec_rad, smoothed, cmap=cmap, norm=norm,
                         shading="nearest", rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.8, orientation="vertical")
    cbar.set_label(
        r'$\left\langle\ \mathrm{median}_{\mathrm{DM}}\left(\Sigma_{f>5}\ P_f\right)\ \right\rangle_{T_{\mathrm{exp}}}$',
        fontsize=13,
    )
    ax.set_title('Skymap of Rednoise Across CHAMPSS Observing Period', fontsize=20, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f'Skymap saved to {output_path}', flush=True)
    else:
        plt.show()

    plt.close(fig)


def plot_coverage(ra, dec, dpi=150, output_path=None):
    '''
    This function plots a dot on our Mollweide axes for each pointing with data.

    Inputs:
    -------
        ra (arr)
        dec (arr)
        dpi (float): figure resolution, used for *both* plt.show() and
            plt.savefig() -- see plot_skymap()'s docstring.
        output_path (str): optional path to save image if desired
    '''
    fig, ax = _setup_axes(dpi=dpi)

    ra_moll = ra_deg_to_moll_rad(ra)
    dec_moll = dec_deg_to_moll_rad(dec)

    ax.scatter(ra_moll, dec_moll, c="hotpink",
               s=1.5, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(f'Pointing Coverage Map', fontsize=20, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f'Coverage map saved to {output_path}', flush=True)
    else:
        plt.show()

    plt.close(fig)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("npz_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("h5_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--smooth-deg", type=float, default=1.0, show_default=True,
              help="Gaussian kernel FWHM, in degrees, for the real-space "
                   "distance-weighted average at each display cell.")
@click.option("--display-res", "display_res", type=float, default=0.25, show_default=True,
              help="Regular lon/dec grid spacing, in degrees, that the kernel "
                   "regression is evaluated on for the Mollweide plot.")
@click.option("--mask-radius-deg", "mask_radius_deg", type=float, default=None,
              help="Display cells farther than this many degrees (great-circle) "
                   "from the nearest actual pointing are always left blank. "
                   "Defaults to 2 * --smooth-deg (about where a Gaussian's "
                   "weight is already down to ~1.5e-5 of its peak) -- lower "
                   "it to blank out more of a coverage island's tapering "
                   "edge, raise it to extend the displayed edge further.")
@click.option("--dpi", type=float, default=150, show_default=True,
              help="Figure resolution, used for BOTH the interactive plt.show() "
                   "window and a saved file, so the two always look the same. "
                   "A mismatch (e.g. a lower-resolution interactive window than "
                   "a saved file's dpi) can make small, faint, real coverage "
                   "get anti-aliased away on screen while still showing up "
                   "clearly (near-black, at the bottom of the log color scale) "
                   "in a saved file at higher dpi -- same data, just resolved "
                   "differently.")
@click.option("--output-skymap", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save sky map to this file (default: display).")
@click.option("--output-coverage", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save coverage map to this file (default: display).")
def main(npz_file, h5_file, smooth_deg, display_res, mask_radius_deg, dpi,
         output_skymap, output_coverage):
    """
    Plot the CHAMPSS rednoise skymap and coverage map from NPZ_FILE (the
    rednoise_dm_info.npz produced by rednoise_dm_behavior.py) and H5_FILE
    (the combined_medians.h5 produced by combine_rednoise_medians.py, used
    to derive each pointing-day's exposure length via its 'scales' array).
    """
    ra, dec, mean_rn = load_data(npz_file, h5_file)

    print("Plotting sky map...", flush=True)
    plot_skymap(ra, dec, mean_rn, smooth_deg, display_res=display_res,
                mask_radius_deg=mask_radius_deg, dpi=dpi, output_path=output_skymap)

    print("Plotting coverage map...", flush=True)
    plot_coverage(ra, dec, dpi=dpi, output_path=output_coverage)


if __name__ == "__main__":
    main()
