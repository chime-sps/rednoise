#!/usr/bin/env python3
'''
This module reads a rednoise info file and produces the following plots:

    1. Skymap: mean across all days of the sum across all frequency bins > 5 of the
       median of the medians across all DMs. We bin and then smooth with a
       Gaussian kernel of customizable size, then project via Mollweide.
    2. Coverage map: one dot per saved pointing.

The mean across days is weighted by each day's exposure length T_exp, since
T_exp differs from pointing to pointing (it depends on the zenith angle at
transit). T_exp is derived from the "scales" array stored per pointing-day in
the combined medians HDF5 file produced by combine_rednoise_medians.py:

    N_in_power_spectrum = np.sum(scales)
    T_exp = 2 * N_in_power_spectrum * tsamp

"scales" gives, for each (already rebinned) output frequency bin, how many
bins of the original power spectrum were averaged together to make it, so its
sum recovers the total number of bins in the original power spectrum. tsamp
is the raw time-domain sampling interval, imported from chime_sps.

Usage:

    python3 rednoise_skymap.py rednoise_dm_info.npz combined_medians.h5
    python3 rednoise_skymap.py rednoise_dm_info.npz combined_medians.h5 --smooth-deg 1.0
        --output-skymap skymap.png --output-coverage coverage.png

To-Do:
    - gridding is still plain RA/Dec Cartesian binning at the raw bin level
      (only the smoothing step corrects for the RA=180 seam and cos(dec));
      a proper equal-area regridding (e.g. HEALPix) would be more correct.
    - adapt to click instead of argparse
'''

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

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

def _smooth_periodic_dec_aware(filled, weights, dec_grid, smooth_deg,
                                dec_resolution, lon_resolution):
    '''
    Smooth the (n_dec, n_lon) `filled`/`weights` grids from grid_and_smooth(),
    correcting for two artifacts a naive 2D gaussian_filter() has on a plain
    RA/Dec Cartesian grid:

    1. RA is a circle, not a line: the longitude axis wraps at +/-180 deg
       (RA=180), but a plain gaussian_filter() treats the array edges as a
       hard boundary. Two points a few degrees apart that straddle that
       wrap point end up ~360 degrees apart in array-index space and never
       get smoothed together, artificially preserving contrast right at the
       seam. Fixed here by smoothing along longitude with `mode="wrap"`.

    2. A degree of RA covers less true sky the closer you get to the poles
       (a "parallel" of constant Dec has physical circumference scaled by
       cos(dec)), so a fixed-degree kernel blends together a much bigger
       true sky area near the poles than at the equator. Fixed here by
       scaling the longitude smoothing sigma by 1/cos(dec), per Dec row,
       so the kernel represents a roughly constant *true* angular scale
       regardless of declination. Right at the poles this is capped at
       "average over the whole circle at this Dec", which is also the
       correct limit as cos(dec) -> 0 (every RA is the same point there).

    The declination axis is smoothed separately (fixed width, non-periodic
    -- Dec does not wrap), after the longitude pass.

    Inputs:
    -------
        filled (arr): (n_dec, n_lon) grid values, 0 where uncovered
        weights (arr): (n_dec, n_lon) coverage weights (1 where covered, 0 else)
        dec_grid (arr): 1-D array of Dec bin centres in degrees, length n_dec
        smooth_deg (float): smoothing kernel width in true degrees
        dec_resolution (float): Dec bin size in degrees
        lon_resolution (float): longitude bin size in degrees

    Returns:
    --------
        s_filled (arr), s_weights (arr): smoothed versions of the inputs,
            same shape, ready to be divided the same way as before.
    '''
    n_dec, n_lon = filled.shape

    # Declination-dependent longitude sigma, in pixels, scaled by 1/cos(dec)
    # so the kernel covers a constant true angular scale. Capped at half the
    # full circle: beyond that, further growth has no meaningful effect
    # (and would blow up the convolution cost), and "average over the whole
    # circle" is exactly correct right at the poles anyway.
    cos_dec = np.cos(np.deg2rad(dec_grid))
    sigma_lon_px_per_row = np.minimum(
        (smooth_deg / lon_resolution) / np.maximum(np.abs(cos_dec), 1e-6),
        n_lon / 2.0,
    )

    # Pass 1: per-row longitude smoothing, periodic (wrap), variable sigma.
    h_filled = np.empty_like(filled)
    h_weights = np.empty_like(weights)
    for i in range(n_dec):
        sigma = sigma_lon_px_per_row[i]
        h_filled[i] = gaussian_filter1d(filled[i], sigma=sigma, mode="wrap")
        h_weights[i] = gaussian_filter1d(weights[i], sigma=sigma, mode="wrap")

    # Pass 2: declination smoothing, fixed width, non-periodic (Dec is
    # bounded, not a circle -- there's no wrap-around at the poles).
    sigma_dec_px = smooth_deg / dec_resolution
    s_filled = gaussian_filter1d(h_filled, sigma=sigma_dec_px, axis=0, mode="nearest")
    s_weights = gaussian_filter1d(h_weights, sigma=sigma_dec_px, axis=0, mode="nearest")

    return s_filled, s_weights


def grid_and_smooth(ra, dec, values, smooth_deg, grid_resolution=0.1):
    '''
    This function bins and then smoothes the mean values across all pointings.

    Binning and smoothing both account for RA/Dec being a sphere, not a flat
    Cartesian plane: longitude is gridded over the full circle and smoothed
    with a periodic (wrapped) kernel, so the RA=180 deg seam doesn't
    artificially block smoothing across it, and the longitude smoothing
    width is scaled by 1/cos(dec) per row, so a fixed *true* angular scale
    is used regardless of declination instead of a fixed number of RA
    degrees (which cover much less true sky near the poles). See
    _smooth_periodic_dec_aware() for details. This is still an approximation
    (not a proper equal-area/HEALPix regridding), but it removes the
    dynamic-range artifacts a plain 2D Cartesian gaussian_filter() has at
    the RA seam and near the poles.

    Note that gridding is done in longitude degrees (lon = -(ra % 360), wrapped to
    (-180, 180]), not in raw RA degrees. Otherwise pcolormesh throws a hissy fit.

    Inputs:
    -------
        ra (arr): 1D array of RAs
        dec (arr): 1D array of Decs
        values (arr): mean_rn from load_data()
        smooth_deg (float): how many degrees to smooth by
        grid_resolution (float): how many degrees to bin by

    Returns:
    --------
        lon_grid (arr): 1-D array of longitude bin centres in degrees, monotonically
            increasing from negative (east) to positive (west), spanning the
            full circle (RA is periodic).
        dec_grid (arr): 1-D array of Dec bin centres in degrees
        smoothed (arr): 2-D array (n_dec, n_lon) of mean of sums of median etc at each pointing,
            NaN in southern sky
    '''

    # convert RA to longitude
    # increases to the left
    # then we wrap to (-180, 180]
    lon = -(ra % 360.0)
    lon = (lon + 180.0) % 360.0 - 180.0

    # Longitude is periodic (RA wraps at 360 deg), so grid it over the full
    # circle rather than just the data's bounding box -- otherwise points
    # straddling the RA=180 seam land at opposite ends of the array and
    # can never be smoothed together correctly. n_lon is chosen so the grid
    # tiles the circle exactly (no gap/overlap at the wrap point).
    n_lon = max(int(round(360.0 / grid_resolution)), 1)
    lon_resolution = 360.0 / n_lon
    lon_grid = -180.0 + lon_resolution * np.arange(n_lon)

    # shouldn't be any weird data past +90 dec but just to be sure
    dec_min, dec_max = max(dec.min() - 1, -90.0), min(dec.max() + 1, 90.0)
    dec_grid = np.arange(dec_min, dec_max + grid_resolution, grid_resolution)

    n_dec = len(dec_grid)

    # Longitude is periodic, so wrap the column index (mod n_lon) instead of
    # clipping it -- lon=180 and lon=-180 are the same point on the sky.
    col = np.mod(np.round((lon - lon_grid[0]) / lon_resolution).astype(int), n_lon)
    row = np.clip(np.round((dec - dec_min) / grid_resolution).astype(int), 0, n_dec - 1)

    grid = np.full((n_dec, n_lon), np.nan)
    count = np.zeros((n_dec, n_lon))
    accum = np.zeros((n_dec, n_lon))

    for i in range(len(ra)):
        r, c = row[i], col[i]
        accum[r, c] += values[i]
        count[r, c] += 1

    mask = count > 0
    grid[mask] = accum[mask] / count[mask]

    # now we apply a Gaussian filter! (periodic in longitude, 1/cos(dec)-scaled
    # width -- see _smooth_periodic_dec_aware())
    filled = np.where(np.isfinite(grid), grid, 0.0)
    weights = np.where(np.isfinite(grid), 1.0, 0.0)

    s_filled, s_weights = _smooth_periodic_dec_aware(
        filled, weights, dec_grid, smooth_deg, grid_resolution, lon_resolution
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(s_weights > 1e-6, s_filled / s_weights, np.nan)

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

def _setup_axes():
    '''
    This function sets up the figure and the Mollweide axes!

    Inputs:
    -------
        unmapped_color (str): color of unmapped regions of the sky
    '''
    fig = plt.figure(figsize=(14, 7))
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


def plot_skymap(ra, dec, mean_rn, smooth_deg, grid_res, output_path=None):
    '''
    This function creates and then plots the rednoise skymap.

    Inputs:
    ------
        ra (arr)
        dec (arr)
        mean_rn (arr)
        smooth_deg (arr)
        grid_res (arr)
        output_path (str): optional path to save image if desired
    '''
    print("Gridding and smoothing ...", flush=True)
    lon_grid, dec_grid, smoothed = grid_and_smooth(ra, dec, mean_rn, smooth_deg,
                                                    grid_resolution=grid_res)

    fig, ax = _setup_axes()

    finite = smoothed[np.isfinite(smoothed)]
    finite_pos = finite[finite > 0]

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
    cbar.set_label('Exposure-weighted mean over time, median across DM, and sum across frequency', fontsize=11)
    ax.set_title('Skymap of Rednoise Across CHAMPSS Observing Period', fontsize=20, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f'Skymap saved to {output_path}', flush=True)
    else:
        plt.show()

    plt.close(fig)


def plot_coverage(ra, dec, output_path=None):
    '''
    This function plots a dot on our Mollweide axes for each pointing with data.
    '''
    fig, ax = _setup_axes()

    ra_moll = ra_deg_to_moll_rad(ra)
    dec_moll = dec_deg_to_moll_rad(dec)

    ax.scatter(ra_moll, dec_moll, c="hotpink",
               s=1.5, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(f'Pointing Coverage Map', fontsize=20, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f'Coverage map saved to {output_path}', flush=True)
    else:
        plt.show()

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_file",
                         help="Path to rednoise_dm_info.npz")
    parser.add_argument("h5_file",
                         help="Path to the combined medians HDF5 file (combined_medians.h5) "
                              "produced by combine_rednoise_medians.py; used to derive each "
                              "pointing-day's exposure length via its 'scales' array.")
    parser.add_argument("--smooth-deg", type=float, default=1.0,
                         help="Gaussian smoothing kernel in degrees (default: 1.0)")
    parser.add_argument("--grid-res", type=float, default=0.1,
                         help="Grid pixel size in degrees (default: 0.1)")
    parser.add_argument("--output-skymap", type=str, default=None,
                         help="Save sky map to this file (default: display)")
    parser.add_argument("--output-coverage", type=str, default=None,
                         help="Save coverage map to this file (default: display)")

    args = parser.parse_args()

    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        sys.exit(f"File not found: {npz_path}")

    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        sys.exit(f"File not found: {h5_path}")

    ra, dec, mean_rn = load_data(npz_path, h5_path)

    print("Plotting sky map...", flush=True)
    plot_skymap(ra, dec, mean_rn, args.smooth_deg, args.grid_res,
                output_path=args.output_skymap)

    print("Plotting coverage map...", flush=True)
    plot_coverage(ra, dec, output_path=args.output_coverage)


if __name__ == "__main__":
    main()
