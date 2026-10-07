#!/usr/bin/env python3
'''
This module reads a rednoise DM median file + a full rednoise file
and produces the following plots:

    1. Rednoise skymap: exposure time-weighted mean over Ndays, median over DM, and
    sum over f > 0.05 Hz of rednoise power
    2. Coverage skymap: one dot per pointing to show current coverage of info

NOTE FOR ROBERT AND LARS:

    Current exp derivation...
    T_exp = length * TSAMP
    (length comes from the beamformer pointings map now, not the padded scales array.
    pointings before Feb 27 2026 use pointings_map_v1-3.json, on/after use pointings_map_v2-0.json)

Usage:

    python3 rednoise_skymap.py rednoise_dm_info.npz
    python3 rednoise_skymap.py rednoise_dm_info.npz --smooth-deg 1.0
        --output-skymap skymap.png --output-coverage coverage.png
'''

import csv
import json
from pathlib import Path

import click
import numpy as np
from scipy.spatial import cKDTree

# my obsessive matplotlib formatting:
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'monospace'
from matplotlib.colors import Normalize, LogNorm

from sps_common.constants import TSAMP

POINTINGS_MAP_CUTOVER = 20260227  # v1-3 before this date, v2-0 on/after
POINTINGS_MAP_DIR = Path(__file__).resolve().parent / "data"
POINTINGS_MAP_V1_3_PATH = POINTINGS_MAP_DIR / "pointings_map_v1-3.json"
POINTINGS_MAP_V2_0_PATH = POINTINGS_MAP_DIR / "pointings_map_v2-0.json"
BRIGHT_SOURCES_PATH = Path(__file__).resolve().parent / data/ "bright_sources.csv"
MAX_MATCH_SEP_DEG = 0.1
RN_FREQ_MIN_HZ = 0.05
NYQUIST_FREQ_HZ = 1.0 / (2.0 * TSAMP)


def _radec_to_xyz(ra, dec):
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(np.mod(ra, 360.0))
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


# -------------
# load our data!
# -------------

def load_pointing_map_tree(pointings_map_path: Path):
    """
    From pointings_map_path, build a nearest-neighbor tree over its
    pointings, for matching against RA/Dec that don't line up with the
    map's grid to exact decimal places.

    Inputs:
    -------
        pointings_map_path (Path): path to a pointings_map_*.json file
            (chime-sps/champss_software beamformer)

    Returns:
    --------
        tree (cKDTree): nearest-neighbor tree, in 3D unit-vector space,
            over the map's (RA, Dec)
        length (arr): "length" field, same order as tree's points
        nchans (arr): "nchans" field, same order as tree's points
    """
    print(f"Loading pointing map {pointings_map_path} ...", flush=True)

    with open(pointings_map_path, "r") as f:
        pointings = json.load(f)

    ra = np.array([p["ra"] for p in pointings])
    dec = np.array([p["dec"] for p in pointings])
    length = np.array([p["length"] for p in pointings], dtype=np.float64)
    nchans = np.array([p["nchans"] for p in pointings], dtype=np.float64)

    tree = cKDTree(_radec_to_xyz(ra, dec))

    print(f"Built nearest-neighbor tree for {len(pointings)} pointings.", flush=True)

    return tree, length, nchans


def nearest_pointing_values(tree, values, query_ra, query_dec, max_sep_deg=MAX_MATCH_SEP_DEG):
    """
    Look up `values` at the nearest pointing-map entry to each
    (query_ra, query_dec), NaN if the nearest one is farther than
    max_sep_deg away (a real miss, not just decimal-place noise).

    Inputs:
    -------
        tree (cKDTree): from load_pointing_map_tree()
        values (arr): field to look up, same order as tree's points
        query_ra (arr), query_dec (arr): RA/Dec to match against the map
        max_sep_deg (float): reject a match farther than this (great-circle)

    Returns:
    --------
        matched (arr): values[nearest], NaN where nearest is farther than
                        max_sep_deg
    """
    chord, idx = tree.query(_radec_to_xyz(query_ra, query_dec))
    sep_deg = np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))

    matched = values[idx]
    return np.where(sep_deg <= max_sep_deg, matched, np.nan)


def last_valid_bin(median_across_dms):
    """
    Each row's last non-pad frequency bin value. median_across_dms is
    padded out to a fixed width with 0 or -1, so a naive column -1 grab is
    only the real white noise bin for the longest rows -- for shorter rows
    it's pad, not data.

    Inputs:
    -------
        median_across_dms (arr): rows x freq bins

    Returns:
    --------
        last_bin (arr): one value per row, NaN if the whole row is pad
    """
    is_pad = (median_across_dms == 0.0) | (median_across_dms == -1.0)
    col = np.arange(median_across_dms.shape[1])
    last_valid_idx = np.where(is_pad, -1, col).max(axis=1)
    has_valid = last_valid_idx >= 0
    last_bin = np.full(median_across_dms.shape[0], np.nan)
    row_idx = np.nonzero(has_valid)[0]
    last_bin[row_idx] = median_across_dms[row_idx, last_valid_idx[row_idx]]
    return last_bin


def load_bright_sources(csv_path: Path):
    """
    For --bright-sources: load bright radio continuum source names and
    positions from csv_path (name, ra_deg, dec_deg, ... columns -- see
    bright_sources.csv), skipping its leading '#' comment lines.

    Inputs:
    -------
        csv_path (Path): path to the bright sources CSV

    Returns:
    --------
        names (list of str)
        ra_deg (arr), dec_deg (arr)
    """
    print(f"Loading bright sources from {csv_path} ...", flush=True)
    with open(csv_path, newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        names, ra_deg, dec_deg = [], [], []
        for row in rows:
            names.append(row["name"].strip())
            ra_deg.append(float(row["ra_deg"]))
            dec_deg.append(float(row["dec_deg"]))

    print(f" {len(names)} bright sources loaded.", flush=True)
    return names, np.array(ra_deg), np.array(dec_deg)


def load_data(npz_path: Path, pointings_map_v1_3_path: Path, pointings_map_v2_0_path: Path,
              nchan_weight: bool = False, normalize: bool = False, plot_whitenoise: bool = False):
    """
    This function loads the DM info file and returns the relevant contents.

    Columns: RA, Dec, year, month, day, 0 if wonky RFI in first bins / 1 if not, DM where wonky behavior starts
    Note: the DM info files were originally collated to analyze the weird DM features in the first bins.

    Inputs:
    -------
        npz_path (Path): path to the DM info file
        pointings_map_v1_3_path (Path): path to pointings_map_v1-3.json,
            used for pointings from before POINTINGS_MAP_CUTOVER
        pointings_map_v2_0_path (Path): path to pointings_map_v2-0.json,
            used for pointings on/after POINTINGS_MAP_CUTOVER
        nchan_weight (bool): if True, also weight rn_sum by 1/nchan per
            pointing (nchan looked up the same v1-3/v2-0 way as T_exp)
        normalize (bool): if True, divide each row's bin-averaged power
            (summed over freq > RN_FREQ_MIN_HZ, then normalized by the
            number of unpadded bins that went into that sum) by its own
            last frequency bin (the white noise level), so different
            days/pointings are compared relative to their own noise floor
            rather than in absolute power
        plot_whitenoise (bool): if True, plot the white noise level itself
            (each row's last real frequency bin) instead of the averaged
            red noise, still Texp/Ndays-averaged the same way. Takes
            precedence over normalize (dividing the white noise level by
            itself is a no-op).

    Returns:
    --------
        RA (arr) : RA of each pointing
        Dec (arr) : Dec of each pointing
        mean_rn (arr): exposure-weighted mean across days of (per-row)
            mean across unpadded freq bins of median across DMs
    """

    print(f"Loading {npz_path} ...", flush=True)
    data = np.load(npz_path)

    info = data["info"]
    median_across_dms = data["median_across_dms"]
    ra = info[:, 0]
    dec = info[:, 1]
    year = info[:, 2].astype(int)
    month = info[:, 3].astype(int)
    day = info[:, 4].astype(int)

    # "scale" is the per-pointing rebinning info carried through from the
    # combined medians file (see rednoise_dm_behavior.py): median_across_dms's
    # columns are REBINNED bins, where stored bin j is an average over
    # scale[row, j] raw (pre-rebinning) FFT bins. It's -1 padded past each
    # row's valid n_freq entries, same convention as median_across_dms.
    try:
        scale = data["scale"]
    except KeyError:
        raise KeyError(
            f"{npz_path} has no 'scale' array -- it's required to convert "
            f"RN_FREQ_MIN_HZ into a per-row bin cutoff (see load_data())."
        )

    # Each row's raw (pre-rebinning) frequency resolution is
    # df_row = NYQUIST_FREQ_HZ / Nbins_row, where Nbins_row is the sum of
    # that row's valid scale entries (the total raw bin count before
    # rebinning) and NYQUIST_FREQ_HZ is a fixed instrumental constant (same
    # for every row -- it depends only on the raw sample rate). The first
    # stored bin is 0 Hz; stored bin j starts cumsum(scale[:j]) raw bins in,
    # i.e. at frequency cumsum(scale[:j]) * df_row.
    is_pad_scale = (scale == -1.0)
    scale_filled = np.where(is_pad_scale, 0.0, scale)
    n_bins_raw = np.sum(scale_filled, axis=1)

    n_missing_scale = np.sum(n_bins_raw <= 0)
    if n_missing_scale:
        print(f"{n_missing_scale} row(s) have no valid 'scale' entries "
              f"(can't locate RN_FREQ_MIN_HZ -- will be dropped).", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        df_raw = NYQUIST_FREQ_HZ / n_bins_raw  # (N,) Hz per raw bin, per row

    cum_raw_bins = np.cumsum(scale_filled, axis=1) - scale_filled  # exclusive cumsum
    freq_at_bin = cum_raw_bins * df_raw[:, None]  # (N, max_n_freq) Hz

    # sum (and count) of only the real, unpadded bins at frequencies above
    # RN_FREQ_MIN_HZ. median_across_dms is padded out to a fixed width with
    # 0 or -1, so a plain sum lets padding corrupt it, and a naive bin count
    # overcounts for any row shorter than the longest one -- mask the pad
    # out of both, on top of the frequency cutoff above.
    is_pad = (median_across_dms == 0.0) | (median_across_dms == -1.0)
    valid_mask = (freq_at_bin > RN_FREQ_MIN_HZ) & ~is_pad
    n_valid_bins = np.sum(valid_mask, axis=1)
    rn_sum = np.sum(np.where(valid_mask, median_across_dms, 0.0), axis=1)

    if plot_whitenoise:
        # plot the white noise level itself (each row's last real freq
        # bin) instead of the averaged red noise -- it still goes through
        # the same Texp/Ndays-weighted averaging below.
        rn_sum = last_valid_bin(median_across_dms)
    elif normalize:
        # normalize the summed power by the number of bins that went into
        # it, then divide by each row's own white noise level (last freq
        # bin).
        last_bin = last_valid_bin(median_across_dms)
        with np.errstate(divide="ignore", invalid="ignore"):
            rn_sum = (rn_sum / n_valid_bins) / last_bin
    else:
        # always normalize the summed power by how many unpadded bins
        # actually went into it, so rows of differing real length are
        # comparable (this used to be a bare sum).
        with np.errstate(divide="ignore", invalid="ignore"):
            rn_sum = rn_sum / n_valid_bins

    # check for data quality
    bad_rn = ~np.isfinite(rn_sum)
    n_bad_rn = np.sum(bad_rn)
    if n_bad_rn:
        print(
            f"{n_bad_rn}/{len(rn_sum)} row(s) had a non-finite "
            f"median_across_dms sum and are being dropped.",
            flush=True,
        )
        keep = ~bad_rn
        ra, dec = ra[keep], dec[keep]
        year, month, day = year[keep], month[keep], day[keep]
        rn_sum = rn_sum[keep]

    tree_v1_3, length_v1_3, nchans_v1_3 = load_pointing_map_tree(pointings_map_v1_3_path)
    tree_v2_0, length_v2_0, nchans_v2_0 = load_pointing_map_tree(pointings_map_v2_0_path)

    row_date = year * 10000 + month * 100 + day
    before_cutover = row_date < POINTINGS_MAP_CUTOVER

    # if/else on the pointing's date: v1-3 before the cutover, v2-0 on/after
    length_matched_v1_3 = nearest_pointing_values(tree_v1_3, length_v1_3, ra, dec)
    length_matched_v2_0 = nearest_pointing_values(tree_v2_0, length_v2_0, ra, dec)
    t_exp = np.where(before_cutover, length_matched_v1_3, length_matched_v2_0) * TSAMP

    n_missing = np.sum(np.isnan(t_exp))
    if n_missing:
        print(f"{n_missing} pointings not in the Texp dict.", flush=True)
        t_exp = np.where(np.isnan(t_exp), 1.0, t_exp)

    if nchan_weight:
        nchan_matched_v1_3 = nearest_pointing_values(tree_v1_3, nchans_v1_3, ra, dec)
        nchan_matched_v2_0 = nearest_pointing_values(tree_v2_0, nchans_v2_0, ra, dec)
        nchan = np.where(before_cutover, nchan_matched_v1_3, nchan_matched_v2_0)

        n_missing_nchan = np.sum(np.isnan(nchan))
        if n_missing_nchan:
            print(f"{n_missing_nchan} pointings not in the nchan dict.", flush=True)
            nchan = np.where(np.isnan(nchan), 1.0, nchan)

        rn_sum = rn_sum / nchan

    pointing_keys = np.round(ra, 4) * 1000 + np.round(dec, 4)
    unique_keys, inv = np.unique(pointing_keys, return_inverse=True)
    n_pointings = len(unique_keys)

    mean_rn = np.zeros(n_pointings, dtype=np.float64)
    weight_sum = np.zeros(n_pointings, dtype=np.float64)
    count = np.zeros(n_pointings, dtype=np.int32)
    ra_pt = np.zeros(n_pointings, dtype=np.float64)   # basically the mean RA across near matches
    dec_pt = np.zeros(n_pointings, dtype=np.float64)  # same here

    # weight each day's contribution to its pointing's mean by that day's T_exp
    # wrong?
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
    Gaussian kernel-weighted sum:
        sum_i( w_i * values_i ) / sum_i( w_i ),   w_i = exp(-0.5 * (d_i / sigma)^2)

    where sigma = FWHM / sqrt(8*ln2) and FWHM is the smooth_deg input.

    Used to be healpix and that made everything worse so I stopped.

    Inputs:
    -------
        ra (arr): 1D array of RAs
        dec (arr): 1D array of Decs
        values (arr): mean_rn from load_data()
        smooth_deg (float): Gaussian kernel FWHM, in degrees
        display_resolution (float): kernel spacing in degrees
    
    Returns:
    --------
        lon_grid (arr): 1D array of longitude bin centres (east is negative, west is pos)
        dec_grid (arr): 1D array of Dec bin centres in degrees
        smoothed (arr): 2D array of kernel-smoothed rednoise power (over all those axes)
    '''
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    #check for nan
    bad = ~(np.isfinite(ra) & np.isfinite(dec) & np.isfinite(values))
    if np.any(bad):
        print(
            f"grid_and_smooth: dropping {np.sum(bad)}/{len(bad)} "
            f"row(s) with non-finite ra/dec/value.",
            flush=True,
        )
        ra, dec, values = ra[~bad], dec[~bad], values[~bad]

    if mask_radius_deg is None:
        mask_radius_deg = 2.0 * smooth_deg

    sigma_deg = smooth_deg / np.sqrt(8.0 * np.log(2.0))  
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra % 360.0)
    data_xyz = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Gridding is done in longitude degrees (lon = -(ra % 360), wrapped to
    # (-180, 180]), not in raw RA degrees, otherwise pcolormesh throws a
    # hissy fit. It covers the full circle since longitude is periodic.
    n_lon = max(int(round(360.0 / display_resolution)), 1)
    lon_resolution = 360.0 / n_lon
    lon_grid = -180.0 + lon_resolution * np.arange(n_lon)

    # shouldn't be any weird data past +90 dec but just to be sure
    dec_min, dec_max = max(dec.min() - 1, -90.0), min(dec.max() + 1, 90.0)
    dec_grid = np.arange(dec_min, dec_max + display_resolution, display_resolution)

    lon_mesh, dec_mesh = np.meshgrid(lon_grid, dec_grid)
    ra_mesh = np.mod(-lon_mesh, 360.0)
    theta_mesh = np.deg2rad(90.0 - dec_mesh)
    phi_mesh = np.deg2rad(ra_mesh)
    mesh_xyz = np.column_stack([
        np.sin(theta_mesh.ravel()) * np.cos(phi_mesh.ravel()),
        np.sin(theta_mesh.ravel()) * np.sin(phi_mesh.ravel()),
        np.cos(theta_mesh.ravel()),
    ])
    n_cells = mesh_xyz.shape[0]
    weighted_sum = np.zeros(n_cells, dtype=np.float64)
    weight_total = np.zeros(n_cells, dtype=np.float64)

    max_chord = 2.0 * np.sin(np.deg2rad(mask_radius_deg) / 2.0)
    sigma_rad = np.deg2rad(sigma_deg)

    grid_tree = cKDTree(mesh_xyz)
    neighbor_lists = grid_tree.query_ball_point(data_xyz, r=max_chord)

    for i, cell_idx in enumerate(neighbor_lists):
        if not cell_idx:
            continue
        cell_idx = np.asarray(cell_idx)
        chord = np.linalg.norm(mesh_xyz[cell_idx] - data_xyz[i], axis=1)
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


# --------
# sun/moon ephemeris, so we can trace their paths over the observation
# (low-precision formulas, no astropy dependency -- good to ~0.01 deg for
# the Sun and ~0.1 deg for the Moon, plenty for an overlay on the skymap)
# --------

def _julian_date(year, month, day, hour=12.0):
    '''
    Calendar date (UTC) -> Julian Date. Defaults to local noon of that date.

    Inputs:
    -------
        year, month, day (arr or scalar)
        hour (float): UTC hour of day to evaluate at

    Returns:
    --------
        jd (arr or scalar)
    '''
    year = np.asarray(year, dtype=np.float64)
    month = np.asarray(month, dtype=np.float64)
    day = np.asarray(day, dtype=np.float64) + hour / 24.0
    y = np.where(month <= 2, year - 1, year)
    m = np.where(month <= 2, month + 12, month)
    a = np.floor(y / 100.0)
    b = 2 - a + np.floor(a / 4.0)
    return np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + day + b - 1524.5


def observation_date_range(npz_path: Path):
    '''
    Daily Julian Date grid (local noon) spanning the first to last day of
    observation recorded in npz_path, for tracing the Sun/Moon over the
    same period.

    Inputs:
    -------
        npz_path (Path): path to the DM info file

    Returns:
    --------
        jd_grid (arr): one JD per day, min day to max day inclusive
    '''
    data = np.load(npz_path)
    info = data["info"]
    year, month, day = info[:, 2].astype(int), info[:, 3].astype(int), info[:, 4].astype(int)
    jd = _julian_date(year, month, day)

    # Sun/Moon are only traced across whatever span is actually in this
    # file -- print it so it's obvious when that's less than a full year
    # (i.e. the Sun's path will be a partial arc, not a closed loop).
    i_min, i_max = np.argmin(jd), np.argmax(jd)
    print(f" Observations span {year[i_min]}-{month[i_min]:02d}-{day[i_min]:02d} to "
          f"{year[i_max]}-{month[i_max]:02d}-{day[i_max]:02d} ({jd.max() - jd.min():.0f} days) "
          f"-- tracing Sun/Moon over that range.", flush=True)

    return np.arange(np.floor(jd.min()), np.floor(jd.max()) + 1.0) + 0.5


def sun_radec(jd):
    '''
    Geocentric apparent RA/Dec of the Sun (low-precision, Meeus ch. 25
    abbreviated formula -- good to about 0.01 deg).

    Inputs:
    -------
        jd (arr): Julian Date(s)

    Returns:
    --------
        ra_deg (arr), dec_deg (arr)
    '''
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T**2) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T**2) % 360.0
    Mr = np.deg2rad(M)
    C = ((1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(Mr)
         + (0.019993 - 0.000101 * T) * np.sin(2 * Mr)
         + 0.000289 * np.sin(3 * Mr))
    true_lon = (L0 + C) % 360.0
    epsilon = 23.439291 - 0.0130042 * T - 0.00000016 * T**2 + 0.000000504 * T**3

    lon_r, eps_r = np.deg2rad(true_lon), np.deg2rad(epsilon)
    ra = np.rad2deg(np.arctan2(np.cos(eps_r) * np.sin(lon_r), np.cos(lon_r))) % 360.0
    dec = np.rad2deg(np.arcsin(np.sin(eps_r) * np.sin(lon_r)))
    return ra, dec


def moon_radec(jd):
    '''
    Geocentric RA/Dec of the Moon (low-precision Keplerian orbit + main
    perturbation terms, per Paul Schlyter's "How to compute planetary
    positions" -- good to about 0.1-0.3 deg, plenty for an overlay here).

    Inputs:
    -------
        jd (arr): Julian Date(s)

    Returns:
    --------
        ra_deg (arr), dec_deg (arr)
    '''
    d = jd - 2451543.5  # days since this algorithm's own epoch (~J2000)

    # Moon's orbital elements (deg)
    N = (125.1228 - 0.0529538083 * d) % 360.0
    i = 5.1454
    w = (318.0634 + 0.1643573223 * d) % 360.0
    a = 60.2666  # earth radii
    e = 0.054900
    M = (115.3654 + 13.0649929509 * d) % 360.0

    # Sun's mean elements, needed for the Moon's perturbation terms below
    Ms = (356.0470 + 0.9856002585 * d) % 360.0
    ws = (282.9404 + 0.0000470935 * d) % 360.0
    Ls = (Ms + ws) % 360.0

    E = M + np.rad2deg(e) * np.sin(np.deg2rad(M)) * (1 + e * np.cos(np.deg2rad(M)))
    for _ in range(4):
        Er = np.deg2rad(E)
        E = E - (E - np.rad2deg(e) * np.sin(Er) - M) / (1 - e * np.cos(Er))
    Er = np.deg2rad(E)

    xv = a * (np.cos(Er) - e)
    yv = a * (np.sqrt(1 - e**2) * np.sin(Er))
    v = np.rad2deg(np.arctan2(yv, xv)) % 360.0
    r = np.sqrt(xv**2 + yv**2)

    Nr, ir = np.deg2rad(N), np.deg2rad(i)
    vwr = np.deg2rad((v + w) % 360.0)
    xh = r * (np.cos(Nr) * np.cos(vwr) - np.sin(Nr) * np.sin(vwr) * np.cos(ir))
    yh = r * (np.sin(Nr) * np.cos(vwr) + np.cos(Nr) * np.sin(vwr) * np.cos(ir))
    zh = r * (np.sin(vwr) * np.sin(ir))

    lon = np.rad2deg(np.arctan2(yh, xh)) % 360.0
    lat = np.rad2deg(np.arctan2(zh, np.sqrt(xh**2 + yh**2)))

    Lm = (N + w + M) % 360.0  # moon's mean longitude
    D = (Lm - Ls) % 360.0     # moon's mean elongation from the sun
    F = (Lm - N) % 360.0      # moon's argument of latitude

    Mr_, Msr, Dr, Fr = np.deg2rad(M), np.deg2rad(Ms), np.deg2rad(D), np.deg2rad(F)
    dlon = (
        -1.274 * np.sin(Mr_ - 2 * Dr)
        + 0.658 * np.sin(2 * Dr)
        - 0.186 * np.sin(Msr)
        - 0.059 * np.sin(2 * Mr_ - 2 * Dr)
        - 0.057 * np.sin(Mr_ - 2 * Dr + Msr)
        + 0.053 * np.sin(Mr_ + 2 * Dr)
        + 0.046 * np.sin(2 * Dr - Msr)
        + 0.041 * np.sin(Mr_ - Msr)
        - 0.035 * np.sin(Dr)
        - 0.031 * np.sin(Mr_ + Msr)
        - 0.015 * np.sin(2 * Fr - 2 * Dr)
        + 0.011 * np.sin(Mr_ - 4 * Dr)
    )
    dlat = (
        -0.173 * np.sin(Fr - 2 * Dr)
        - 0.055 * np.sin(Mr_ - Fr - 2 * Dr)
        - 0.046 * np.sin(Mr_ + Fr - 2 * Dr)
        + 0.033 * np.sin(Fr + 2 * Dr)
        + 0.017 * np.sin(2 * Mr_ + Fr)
    )
    lon = (lon + dlon) % 360.0
    lat = lat + dlat

    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    eps_r = np.deg2rad(23.439291 - 0.0130042 * ((jd - 2451545.0) / 36525.0))

    xg = np.cos(lon_r) * np.cos(lat_r)
    yg = np.sin(lon_r) * np.cos(lat_r)
    zg = np.sin(lat_r)

    xe = xg
    ye = yg * np.cos(eps_r) - zg * np.sin(eps_r)
    ze = yg * np.sin(eps_r) + zg * np.cos(eps_r)

    ra = np.rad2deg(np.arctan2(ye, xe)) % 360.0
    dec = np.rad2deg(np.arcsin(ze))
    return ra, dec


MOON_SIDEREAL_MONTH_DAYS = 27.321661  # Moon's orbit around us, not the synodic (phase) month


def moon_average_path(jd_grid, n_phase_bins=100):
    '''
    The Moon retraces almost the same RA/Dec loop every sidereal month, so
    plotting its daily position over a multi-month observation just draws
    the same curve over and over. This folds every day in jd_grid onto one
    representative sidereal month (by day-of-cycle) and averages RA/Dec
    within each phase bin, so a long observation still traces a single,
    clean "average" Moon path instead of an overplotted smear.

    Inputs:
    -------
        jd_grid (arr): Julian Dates to average over (e.g. from
            observation_date_range())
        n_phase_bins (int): number of bins across one sidereal month

    Returns:
    --------
        ra_deg (arr), dec_deg (arr): one averaged point per populated phase
            bin, in phase order (so it traces out as a single loop)
    '''
    ra, dec = moon_radec(jd_grid)
    phase = np.mod(jd_grid, MOON_SIDEREAL_MONTH_DAYS)
    bin_edges = np.linspace(0.0, MOON_SIDEREAL_MONTH_DAYS, n_phase_bins + 1)
    bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_phase_bins - 1)

    # RA wraps at 360 deg, so average it as a unit vector (circular mean),
    # not arithmetically -- Dec never wraps, so a plain mean is fine there.
    ra_r = np.deg2rad(ra)
    sin_sum = np.zeros(n_phase_bins)
    cos_sum = np.zeros(n_phase_bins)
    dec_sum = np.zeros(n_phase_bins)
    count = np.zeros(n_phase_bins)
    np.add.at(sin_sum, bin_idx, np.sin(ra_r))
    np.add.at(cos_sum, bin_idx, np.cos(ra_r))
    np.add.at(dec_sum, bin_idx, dec)
    np.add.at(count, bin_idx, 1)

    has_data = count > 0
    avg_ra = np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360.0
    avg_dec = dec_sum / np.maximum(count, 1)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    order = np.argsort(bin_centers)
    order = order[has_data[order]]

    # close the loop back to the first point, since the average path is
    # periodic over one sidereal month
    return avg_ra[np.append(order, order[0])], avg_dec[np.append(order, order[0])]


# -------
# plotting stuff
# -------

def _plot_radec_path(ax, ra_deg, dec_deg, **kwargs):
    '''
    Plot a path of RA/Dec points (e.g. the Sun or Moon's position) on our
    Mollweide axes as a single line/legend entry, inserting a gap (NaN)
    wherever it crosses the RA=0/360 seam so we don't draw a spurious
    chord across the whole plot.
    '''
    lon_rad = ra_deg_to_moll_rad(ra_deg)
    dec_rad = dec_deg_to_moll_rad(dec_deg)
    seam = np.where(np.abs(np.diff(lon_rad)) > np.pi)[0] + 1
    lon_rad = np.insert(lon_rad, seam, np.nan)
    dec_rad = np.insert(dec_rad, seam, np.nan)
    ax.plot(lon_rad, dec_rad, **kwargs)


def _plot_bright_sources(ax, names, ra_deg, dec_deg, color="cyan", fontsize=10):
    '''
    Mark each bright radio source with a small '+' and its name labeled
    just above it, on our Mollweide axes.
    '''
    lon_rad = ra_deg_to_moll_rad(np.asarray(ra_deg))
    dec_rad = dec_deg_to_moll_rad(np.asarray(dec_deg))
    ax.scatter(lon_rad, dec_rad, marker="+", s=50, linewidths=1.3, color=color, zorder=6)

    label_offset_rad = np.deg2rad(2.5)
    for lon, dec, name in zip(lon_rad, dec_rad, names):
        ax.text(lon, dec + label_offset_rad, name, ha="center", va="bottom",
                fontsize=fontsize, color=color, zorder=6)


def _setup_axes(dpi=150):
    '''
    This function sets up the figure and the Mollweide axes!
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
    label_dec_rad = np.deg2rad(-15)
    for lon_rad, label in zip(hour_ticks_rad, ra_hour_labels):
        ax.text(lon_rad, label_dec_rad, f"{label:.0f}h",
                ha="center", va="top", fontsize=17, color="black")

    ax.set_xlabel("Right Ascension", labelpad=14, fontsize=16)
    ax.set_ylabel("Declination (deg)", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)

    return fig, ax


def plot_skymap(ra, dec, mean_rn, smooth_deg, display_res=0.25,
                 mask_radius_deg=None, dpi=150, nchan_weight=False, plot_whitenoise=False,
                 normalize=False, title=None, sun_path=None, moon_path=None,
                 bright_sources=None, output_path=None):
    '''
    This function creates and then plots the rednoise skymap.

    Inputs:
    ------
        ra (arr)
        dec (arr)
        mean_rn (arr)
        smooth_deg (float): Gaussian kernel FWHM in degrees
        display_res (float): regular lon/dec display-grid spacing in degrees
        nchan_weight (bool): if True, note the 1/nchan weighting on the colorbar label
        plot_whitenoise (bool): if True, label as the white noise level instead
            of the bin-averaged red noise (mean_rn itself should already be
            computed accordingly by load_data)
        normalize (bool): if True, label as white-noise-divided (mean_rn
            itself should already be computed accordingly by load_data)
        title (str): optional override for the plot title
        sun_path (tuple): optional (ra_deg, dec_deg) of the Sun over the
            observation, traced as a dashed line
        moon_path (tuple): same, for the Moon
        bright_sources (tuple): optional (names, ra_deg, dec_deg) of bright
            radio sources, marked and labeled just above their position
        output_path (str): optional path to save image if desired
    '''
    print("Gridding and smoothing ...", flush=True)
    lon_grid, dec_grid, smoothed = grid_and_smooth(ra, dec, mean_rn, smooth_deg,
                                                    display_resolution=display_res,
                                                    mask_radius_deg=mask_radius_deg)

    fig, ax = _setup_axes(dpi=dpi)

    finite = smoothed[np.isfinite(smoothed)]
    finite_pos = finite[finite > 0]

    norm = LogNorm(vmin=np.percentile(finite_pos, 2),
                    vmax=np.percentile(finite_pos, 98))

    # convert to radians for pcolormesh... could just save as radians initially
    # maybe easier
    lon_rad = np.deg2rad(lon_grid)
    dec_rad = np.deg2rad(dec_grid)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad("white")

    pcm = ax.pcolormesh(lon_rad, dec_rad, smoothed, cmap=cmap, norm=norm,
                         shading="nearest", rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.8, orientation="vertical")
    if plot_whitenoise:
        cbar_label_body = r'\left\langle\ P_{\mathrm{last~bin}}\ \right\rangle_{T_{\mathrm{exp}}}'
    elif normalize:
        cbar_label_body = r'\left\langle\ \mathrm{median}_{\mathrm{DM}}\left(\langle P_f \rangle_{f>0.05\,\mathrm{Hz}} / P_{\mathrm{last~bin}}\right)\ \right\rangle_{T_{\mathrm{exp}}}'
    else:
        cbar_label_body = r'\left\langle\ \mathrm{median}_{\mathrm{DM}}\left(\langle P_f \rangle_{f>0.05\,\mathrm{Hz}}\right)\ \right\rangle_{T_{\mathrm{exp}}}'
    if nchan_weight:
        cbar_label_body += r' / N_{\mathrm{chan}}'
    cbar.set_label(f'${cbar_label_body}$', fontsize=16)

    default_title = ('Skymap of White Noise Across CHAMPSS Observing Period' if plot_whitenoise
                      else 'Skymap of Rednoise Across CHAMPSS Observing Period')
    ax.set_title(title or default_title, fontsize=24, fontweight='bold')

    if sun_path is not None:
        _plot_radec_path(ax, sun_path[0], sun_path[1], linestyle="--", color="gold",
                          linewidth=1.5, zorder=5, label="Sun")
    if moon_path is not None:
        _plot_radec_path(ax, moon_path[0], moon_path[1], linestyle="--", color="silver",
                          linewidth=1.5, zorder=5, label="Moon")
    if sun_path is not None or moon_path is not None:
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8)

    if bright_sources is not None:
        _plot_bright_sources(ax, bright_sources[0], bright_sources[1], bright_sources[2])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f'Skymap saved to {output_path}', flush=True)
    else:
        plt.show()

    plt.close(fig)


def plot_coverage(ra, dec, dpi=150, sun_path=None, moon_path=None, bright_sources=None,
                   output_path=None):
    '''
    This function plots a dot on our Mollweide axes for each pointing with data.

    Inputs:
    -------
        ra (arr)
        dec (arr)
        sun_path (tuple): optional (ra_deg, dec_deg) of the Sun over the
            observation, traced as a dashed line
        moon_path (tuple): same, for the Moon
        bright_sources (tuple): optional (names, ra_deg, dec_deg) of bright
            radio sources, marked and labeled just above their position
        output_path (str): optional path to save image if desired
    '''
    fig, ax = _setup_axes(dpi=dpi)

    ra_moll = ra_deg_to_moll_rad(ra)
    dec_moll = dec_deg_to_moll_rad(dec)

    ax.scatter(ra_moll, dec_moll, c="hotpink",
               s=1.5, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(f'Pointing Coverage Map', fontsize=24, fontweight='bold')

    if sun_path is not None:
        _plot_radec_path(ax, sun_path[0], sun_path[1], linestyle="--", color="gold",
                          linewidth=1.5, zorder=5, label="Sun")
    if moon_path is not None:
        _plot_radec_path(ax, moon_path[0], moon_path[1], linestyle="--", color="silver",
                          linewidth=1.5, zorder=5, label="Moon")
    if sun_path is not None or moon_path is not None:
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8)

    if bright_sources is not None:
        _plot_bright_sources(ax, bright_sources[0], bright_sources[1], bright_sources[2],
                              color="black")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f'Coverage map saved to {output_path}', flush=True)
    else:
        plt.show()

    plt.close(fig)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("npz_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--smooth-deg", type=float, default=1.0, show_default=True,
              help="Gaussian kernel FWHM, in degrees, for the real-space "
                   "distance-weighted average at each display cell.")
@click.option("--display-res", "display_res", type=float, default=0.25, show_default=True,
              help="Regular lon/dec grid spacing, in degrees, that the kernel "
                   "regression is evaluated on for the Mollweide plot.")
@click.option("--mask-radius-deg", "mask_radius_deg", type=float, default=None,
              help="Display cells farther than this many degrees (great-circle) ")
@click.option("--dpi", type=float, default=150, show_default=True,
              help="Figure resolution.")
@click.option("--nchan-weight", "nchan_weight", is_flag=True, default=False,
              help="Weight rednoise values by 1/nchan before plotting.")
@click.option("--normalize", "normalize", is_flag=True, default=False,
              help="Divide each row's bin-averaged power by its own last "
                   "freq bin (white noise level), before averaging across days.")
@click.option("--plot-whitenoise", "plot_whitenoise", is_flag=True, default=False,
              help="Plot the white noise level (each row's last freq bin, "
                   "Texp/Ndays-averaged) instead of the summed red noise. "
                   "Takes precedence over --normalize.")
@click.option("--title", "title", type=str, default=None,
              help="Override the skymap plot title (default depends on "
                   "--plot-whitenoise).")
@click.option("--bright-sources", "bright_sources", is_flag=True, default=False,
              help="Label bright radio continuum sources (from bright_sources.csv "
                   "next to this script) just above their position.")
@click.option("--output-skymap", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save sky map to this file (default: display).")
@click.option("--output-coverage", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save coverage map to this file (default: display).")
def main(npz_file, smooth_deg, display_res, mask_radius_deg, dpi, nchan_weight, normalize,
         plot_whitenoise, title, bright_sources, output_skymap, output_coverage):
    """
    Plot the CHAMPSS rednoise skymap and coverage map from NPZ_FILE (the
    rednoise_dm_info.npz produced by rednoise_dm_behavior.py). Pointing
    exposure lengths come from pointings_map_v1-3.json / pointings_map_v2-0.json
    in scripts/data/ (v1-3 for pointings before Feb 27 2026, v2-0 on/after).
    """
    for p in (POINTINGS_MAP_V1_3_PATH, POINTINGS_MAP_V2_0_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Expected pointings map at {p}")

    ra, dec, mean_rn = load_data(npz_file, POINTINGS_MAP_V1_3_PATH, POINTINGS_MAP_V2_0_PATH,
                                  nchan_weight=nchan_weight, normalize=normalize,
                                  plot_whitenoise=plot_whitenoise)

    print("Tracing Sun/Moon paths over the observation...", flush=True)
    jd_grid = observation_date_range(npz_file)
    sun_path = sun_radec(jd_grid)
    moon_path = moon_average_path(jd_grid)

    bright_source_info = None
    if bright_sources:
        if not BRIGHT_SOURCES_PATH.exists():
            raise FileNotFoundError(f"Expected bright sources CSV at {BRIGHT_SOURCES_PATH}")
        bright_source_info = load_bright_sources(BRIGHT_SOURCES_PATH)

    print("Plotting sky map...", flush=True)
    plot_skymap(ra, dec, mean_rn, smooth_deg, display_res=display_res,
                mask_radius_deg=mask_radius_deg, dpi=dpi, nchan_weight=nchan_weight,
                plot_whitenoise=plot_whitenoise, normalize=normalize, title=title,
                sun_path=sun_path, moon_path=moon_path, bright_sources=bright_source_info,
                output_path=output_skymap)

    print("Plotting coverage map...", flush=True)
    plot_coverage(ra, dec, dpi=dpi, sun_path=sun_path, moon_path=moon_path,
                  bright_sources=bright_source_info, output_path=output_coverage)


if __name__ == "__main__":
    main()
