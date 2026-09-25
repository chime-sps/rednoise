#!/usr/bin/env python3
'''
This module reads a rednoise DM median file + a full rednoise file
and produces the following plots:

    1. Rednoise skymap: exposure time-weighted mean over Ndays, median over DM, and
    sum over f_bin > 5 of rednoise power
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

import json
from pathlib import Path

import click
import numpy as np
from scipy.spatial import cKDTree

# my obsessive matplotlib formatting:
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'monospace'
from matplotlib.colors import Normalize, LogNorm

from sps_common.constants import TSAMP

POINTINGS_MAP_CUTOVER = 20260227  # v1-3 before this date, v2-0 on/after
POINTINGS_MAP_DIR = Path(__file__).resolve().parent / "data"
POINTINGS_MAP_V1_3_PATH = POINTINGS_MAP_DIR / "pointings_map_v1-3.json"
POINTINGS_MAP_V2_0_PATH = POINTINGS_MAP_DIR / "pointings_map_v2-0.json"

# exact (ra,dec) dict matching against the pointings map missed ~everything
# (the map's own grid isn't even self-consistent to 4 decimal places between
# v1-3 and v2-0), so we match nearest-neighbor instead, in 3D unit-vector
# space (no RA wraparound / pole weirdness). 0.1 deg is comfortably above
# the worst v1-3/v2-0 grid drift we've seen (~0.06 deg) and comfortably
# below the spacing between genuinely different beams (~0.25 deg).
MAX_MATCH_SEP_DEG = 0.1


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


def load_data(npz_path: Path, pointings_map_v1_3_path: Path, pointings_map_v2_0_path: Path,
              nchan_weight: bool = False, normalize: bool = False):
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
        normalize (bool): if True, divide each row's median-power-vs-freq
            by its own last frequency bin (the white noise level) before
            summing over freq bins > 5, so different days/pointings are
            compared relative to their own noise floor rather than in
            absolute power

    Returns:
    --------
        RA (arr) : RA of each pointing
        Dec (arr) : Dec of each pointing
        mean_rn (arr): exposure-weighted mean across days of sum across freq bins of median across DMs
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

    # take sum of median of medians across frequency bins after 5
    rn_sum = np.sum(median_across_dms[:, 5:], axis=1)

    if normalize:
        # normalize by each row's own white noise level (last freq bin)
        # before averaging across days -- sum(row[5:]/last) == sum(row[5:])/last
        # since last is a per-row scalar, so this is equivalent to just
        # dividing rn_sum directly.
        last_bin = median_across_dms[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rn_sum = rn_sum / last_bin

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


# -------
# plotting stuff
# -------

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
                ha="center", va="top", fontsize=14, color="black")

    ax.set_xlabel("Right Ascension", labelpad=14, fontsize=13)
    ax.set_ylabel("Declination (deg)", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)

    return fig, ax


def plot_skymap(ra, dec, mean_rn, smooth_deg, display_res=0.25,
                 mask_radius_deg=None, dpi=150, nchan_weight=False, output_path=None):
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
    if nchan_weight:
        cbar_label = r'$\left\langle\ \mathrm{median}_{\mathrm{DM}}\left(\Sigma_{f>5}\ P_f\right)\ \right\rangle_{T_{\mathrm{exp}}} / N_{\mathrm{chan}}$'
    else:
        cbar_label = r'$\left\langle\ \mathrm{median}_{\mathrm{DM}}\left(\Sigma_{f>5}\ P_f\right)\ \right\rangle_{T_{\mathrm{exp}}}$'
    cbar.set_label(cbar_label, fontsize=13)
    ax.set_title('Skymap of Rednoise Across CHAMPSS Observing Period', fontsize=20, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
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
              help="Normalize each row by its own last freq bin (white noise "
                   "level) before averaging across days.")
@click.option("--output-skymap", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save sky map to this file (default: display).")
@click.option("--output-coverage", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save coverage map to this file (default: display).")
def main(npz_file, smooth_deg, display_res, mask_radius_deg, dpi, nchan_weight, normalize,
         output_skymap, output_coverage):
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
                                  nchan_weight=nchan_weight, normalize=normalize)

    print("Plotting sky map...", flush=True)
    plot_skymap(ra, dec, mean_rn, smooth_deg, display_res=display_res,
                mask_radius_deg=mask_radius_deg, dpi=dpi, nchan_weight=nchan_weight,
                output_path=output_skymap)

    print("Plotting coverage map...", flush=True)
    plot_coverage(ra, dec, dpi=dpi, output_path=output_coverage)


if __name__ == "__main__":
    main()
