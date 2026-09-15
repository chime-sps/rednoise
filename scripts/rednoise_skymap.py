#!/usr/bin/env python3
'''
This module reads a rednoise info file and produces the following plots:

    1. Skymap: mean across all days of the sum across all frequency bins > 5 of the
       median of the medians across all DMs. We bin onto a HEALPix grid and then
       smooth with a Gaussian beam of customizable size, then project via Mollweide.
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
import healpy as hp
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
    # real at survey scale (millions of rows), and it matters a lot more
    # here than a normal bad-value would: even a single NaN pointing
    # reaching grid_and_smooth() poisons hp.smoothing()'s spherical-
    # harmonic transform for the *entire* sky (unlike a real-space filter,
    # which would only spoil nearby pixels), silently turning every output
    # value into NaN. Drop those rows here, with the count surfaced, rather
    # than letting them propagate into a confusing failure several
    # functions away.
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

def grid_and_smooth(ra, dec, values, smooth_deg, nside=128, display_resolution=0.25,
                     mask_radius_deg=None):
    '''
    This function bins pointings onto a HEALPix grid and then smoothes the
    mean values across all pointings with a Gaussian beam, before resampling
    onto a regular lon/dec grid for display.

    HEALPix is an equal-area, iso-latitude pixelization of the sphere, so
    (unlike the plain RA/Dec Cartesian binning this replaces) it has neither
    of the two projection artifacts that plain Cartesian binning+smoothing
    has to be specially corrected for:

    1. RA=180 seam: HEALPix pixels and hp.smoothing()'s spherical-harmonic
       beam convolution have no notion of an RA=0/360 boundary at all --
       there is no array edge in longitude for two nearby points to land on
       opposite sides of.
    2. High declination / poles: HEALPix pixels have (approximately) equal
       true sky area everywhere, including near the poles, so a
       fixed-angle Gaussian beam smooths a constant true sky area
       regardless of declination, unlike a fixed-degree-of-RA kernel on a
       Cartesian grid.

    Binning: each pointing's `values` entry is accumulated into its HEALPix
    pixel's running sum, alongside a hit-count map. Smoothing: hp.smoothing()
    (a spherical-harmonic Gaussian beam of FWHM `smooth_deg`) is applied
    separately to the sum map and the count map, so an isolated value is
    still reproduced exactly. Display: the smoothed sum map and count map
    are each resampled (still separately) onto a regular lon/dec grid via
    hp.get_interp_val(), purely so the existing matplotlib Mollweide +
    pcolormesh plotting code (with its custom RA-hour axis ticks) can be
    reused unchanged. Only on that regular grid -- never before -- are they
    divided and masked by coverage (accumulate-then-divide-by-coverage,
    same pattern the old Cartesian version used). This order matters: an
    earlier version of this function divided and coverage-masked (to NaN)
    while still in HEALPix space, then interpolated that NaN-containing map
    for display. Since hp.get_interp_val() bilinearly blends each display
    point's 4 neighboring HEALPix pixels, and a weighted sum touching even
    one NaN neighbor is itself NaN, that poisoned the display grid almost
    everywhere on a real, only-partially-covered sky (most HEALPix pixels
    at typical `nside` are never hit by an actual pointing) -- including
    display cells sitting right on top of real data. See the module-level
    concerns note delivered alongside this function for further tradeoffs
    of the interpolate-then-display-grid design.

    Inputs:
    -------
        ra (arr): 1D array of RAs
        dec (arr): 1D array of Decs
        values (arr): mean_rn from load_data()
        smooth_deg (float): Gaussian beam FWHM, in degrees, to smooth by
        nside (int): HEALPix resolution parameter; must be a power of 2.
            Pixel size is roughly 58.6 / nside degrees. Higher nside means
            finer pixels but a slower hp.smoothing() call.
        display_resolution (float): spacing, in degrees, of the regular
            lon/dec grid the smoothed HEALPix map is resampled onto for
            plotting (independent of `nside`).
        mask_radius_deg (float or None): a display cell farther than this
            great-circle distance from the nearest *actual* input pointing
            is always masked to NaN, regardless of what count_mesh says.
            Defaults to 2 * smooth_deg (two beam widths) if None. This
            guards against a real hp.smoothing() artifact: representing a
            sparse, sharp-edged coverage pattern (real gaps between
            drift-scan pointings) as a truncated spherical-harmonic series
            can ring, occasionally swinging the smoothed count map back
            *above* the coverage threshold in small, isolated spots well
            inside a real gap -- painting spurious, disconnected "filled
            in" dots with real-looking values in regions that have no data
            at all. Real-space distance to actual data can't ring, so it's
            what decides whether a cell is shown; the harmonic-smoothed
            accum/count maps are still what decide the cell's *value*.

    Returns:
    --------
        lon_grid (arr): 1-D array of longitude bin centres in degrees, monotonically
            increasing from negative (east) to positive (west), spanning the
            full circle (RA is periodic).
        dec_grid (arr): 1-D array of Dec bin centres in degrees
        smoothed (arr): 2-D array (n_dec, n_lon) of mean of sums of median etc at each pointing,
            NaN where there is no nearby HEALPix coverage
    '''
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a positive power of 2, got {nside}")

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Defensive guard, independent of whatever the caller already did: a
    # single non-finite ra/dec/value reaching hp.smoothing() below poisons
    # its spherical-harmonic transform for the *entire* sky (every output
    # pixel depends on every input pixel via that transform), not just the
    # pixel it landed in -- silently turning the whole map to NaN. Drop
    # such rows here so this function is safe on its own, whatever the
    # caller's input hygiene looks like.
    bad = ~(np.isfinite(ra) & np.isfinite(dec) & np.isfinite(values))
    if np.any(bad):
        print(
            f"  [WARN] grid_and_smooth: dropping {np.sum(bad)}/{len(bad)} "
            f"row(s) with non-finite ra/dec/value.",
            flush=True,
        )
        ra, dec, values = ra[~bad], dec[~bad], values[~bad]

    # HEALPix uses colatitude (0 at the north pole, pi at the south pole)
    # and longitude in [0, 2*pi), both in radians.
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra % 360.0)

    npix = hp.nside2npix(nside)
    pix = np.atleast_1d(hp.ang2pix(nside, theta, phi))

    accum = np.zeros(npix, dtype=np.float64)
    count = np.zeros(npix, dtype=np.float64)
    np.add.at(accum, pix, values)
    np.add.at(count, pix, 1.0)

    fwhm_rad = np.deg2rad(smooth_deg)
    s_accum = hp.smoothing(accum, fwhm=fwhm_rad)
    s_count = hp.smoothing(count, fwhm=fwhm_rad)

    # NOTE: do NOT build a "divide, then NaN out uncovered pixels" HEALPix
    # map here and interpolate *that*. hp.get_interp_val() does bilinear
    # interpolation across each query point's 4 neighboring pixels, and a
    # weighted sum involving even one NaN neighbor is itself NaN -- so a
    # HEALPix-space map that's NaN at every uncovered pixel poisons the
    # interpolated result at any display point whose neighbor stencil
    # touches even one of them. On a real, only-partially-covered sky (most
    # HEALPix pixels never hit by an actual pointing), that stencil touches
    # an uncovered pixel almost everywhere -- including display cells that
    # sit right on top of real data -- wiping out nearly the entire map.
    # Interpolating the raw (never-NaN) smoothed accum/count maps
    # separately, and only masking by coverage *after* landing on the
    # regular display grid, avoids this entirely.

    # Resample onto a regular lon/dec grid, purely for compatibility with
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

    accum_mesh = hp.get_interp_val(s_accum, theta_mesh, phi_mesh)
    count_mesh = hp.get_interp_val(s_count, theta_mesh, phi_mesh)

    # Coverage mask: real-space great-circle distance to the nearest actual
    # input pointing, NOT the harmonic-smoothed count_mesh. hp.smoothing()'s
    # spherical-harmonic beam can ring on a sparse, sharp-edged coverage
    # pattern (real gaps between drift-scan pointings), occasionally
    # swinging count_mesh back above the threshold in small isolated spots
    # well inside a genuine gap -- painting spurious, disconnected "filled
    # in" dots with plausible-looking values where there's no data at all.
    # A KDTree distance query on the original points can't ring, so it's
    # what decides *whether* a cell is shown; count_mesh/accum_mesh still
    # decide its smoothed *value*.
    if mask_radius_deg is None:
        mask_radius_deg = 2.0 * smooth_deg

    data_xyz = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    mesh_xyz = np.column_stack([
        np.sin(theta_mesh.ravel()) * np.cos(phi_mesh.ravel()),
        np.sin(theta_mesh.ravel()) * np.sin(phi_mesh.ravel()),
        np.cos(theta_mesh.ravel()),
    ])
    chord_dist, _ = cKDTree(data_xyz).query(mesh_xyz, k=1)
    # chord length between two unit vectors separated by angle a is
    # 2*sin(a/2); compare directly in chord space to avoid an arcsin per
    # query point.
    max_chord = 2.0 * np.sin(np.deg2rad(mask_radius_deg) / 2.0)
    near_data = (chord_dist <= max_chord).reshape(theta_mesh.shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(near_data & (count_mesh > 1e-6), accum_mesh / count_mesh, np.nan)

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


def plot_skymap(ra, dec, mean_rn, smooth_deg, nside=128, display_res=0.25,
                 mask_radius_deg=None, output_path=None):
    '''
    This function creates and then plots the rednoise skymap.

    Inputs:
    ------
        ra (arr)
        dec (arr)
        mean_rn (arr)
        smooth_deg (float): Gaussian beam FWHM in degrees
        nside (int): HEALPix resolution parameter (power of 2)
        display_res (float): regular lon/dec display-grid spacing in degrees
        mask_radius_deg (float or None): see grid_and_smooth(); defaults to
            2 * smooth_deg if None
        output_path (str): optional path to save image if desired
    '''
    print("Gridding and smoothing ...", flush=True)
    lon_grid, dec_grid, smoothed = grid_and_smooth(ra, dec, mean_rn, smooth_deg,
                                                    nside=nside,
                                                    display_resolution=display_res,
                                                    mask_radius_deg=mask_radius_deg)

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
    cbar.set_label(
        r'$\left\langle\ \mathrm{median}_{\mathrm{DM}}\left(\Sigma_{f>5}\ P_f\right)\ \right\rangle_{T_{\mathrm{exp}}}$',
        fontsize=13,
    )
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("npz_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("h5_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--smooth-deg", type=float, default=1.0, show_default=True,
              help="Gaussian smoothing beam FWHM, in degrees.")
@click.option("--nside", type=int, default=128, show_default=True,
              help="HEALPix resolution parameter (must be a power of 2); "
                   "pixel size is roughly 58.6/nside degrees.")
@click.option("--display-res", "display_res", type=float, default=0.25, show_default=True,
              help="Regular lon/dec grid spacing, in degrees, that the smoothed "
                   "HEALPix map is resampled onto for the Mollweide plot.")
@click.option("--mask-radius-deg", "mask_radius_deg", type=float, default=None,
              help="Display cells farther than this many degrees (great-circle) "
                   "from the nearest actual pointing are always left blank, "
                   "regardless of smoothed coverage. Defaults to 2 * --smooth-deg. "
                   "Guards against hp.smoothing() ringing on sparse coverage "
                   "'filling in' spurious disconnected dots in real gaps -- "
                   "lower it if real gaps still show spurious fill, raise it if "
                   "genuinely-covered edges are being clipped.")
@click.option("--output-skymap", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save sky map to this file (default: display).")
@click.option("--output-coverage", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save coverage map to this file (default: display).")
def main(npz_file, h5_file, smooth_deg, nside, display_res, mask_radius_deg,
         output_skymap, output_coverage):
    """
    Plot the CHAMPSS rednoise skymap and coverage map from NPZ_FILE (the
    rednoise_dm_info.npz produced by rednoise_dm_behavior.py) and H5_FILE
    (the combined_medians.h5 produced by combine_rednoise_medians.py, used
    to derive each pointing-day's exposure length via its 'scales' array).
    """
    ra, dec, mean_rn = load_data(npz_file, h5_file)

    print("Plotting sky map...", flush=True)
    plot_skymap(ra, dec, mean_rn, smooth_deg, nside=nside, display_res=display_res,
                mask_radius_deg=mask_radius_deg, output_path=output_skymap)

    print("Plotting coverage map...", flush=True)
    plot_coverage(ra, dec, output_path=output_coverage)


if __name__ == "__main__":
    main()
