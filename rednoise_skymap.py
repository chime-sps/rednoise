#!/usr/bin/env python3
'''
This module reads a rednoise info file and produces the following plots:

  1. Skymap: mean across all days of the sum across all frequency bins > 5 of the 
     median of the medians across all DMs. We bin and then smooth with a
     Gaussian kernel of customizable size, then project via Mollweide.

  2. Coverage map: one dot per saved pointing.


Usage:

    python3 plot_rednoise_skymap.py rednoise_dm_info.npz
    python3 plot_rednoise_skymap.py rednoise_dm_info.npz --smooth-deg 1.0
        --output-skymap skymap.png --output-coverage coverage.png

To-Do:
    - figure out better way of gridding other than cartesian...
    - adapt to click instead of argparse
'''

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

#my obsessive matplotlib formatting:
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'monospace' 
from matplotlib.colors import Normalize, LogNorm


#-------------
#load our data!
#-------------

def load_data(npz_path: Path):
    """
    This function loads the DM info file and returns the relevant contents.

    Columns: RA, Dec, year, month, day, 0 if wonky RFI in first bins / 1 if not, DM where wonky behavior starts
    
    Inputs:
    -------
        npz_path (Path): path to the DM info file

    Returns:
    --------
        RA (arr)     : RA of each pointing
        Dec (arr)    : Dec of each pointing
        mean_rn (arr): mean across days of sum across freq bins of median across DMs
    """

    print(f"Loading {npz_path} ...", flush=True)
    data = np.load(npz_path)
    info              = data["info"]               #shape: (N, 7)
    median_across_dms = data["median_across_dms"]  #shape: (N, max_n_freq)

    ra   = info[:, 0]
    dec  = info[:, 1]
    year = info[:, 2].astype(int)

    #take sum of median of medians across frequency bins after 5
    rn_sum = np.sum(median_across_dms[:, 5:], axis=1)   #shape: (N)

    #figure out how many pointings we have
    #consider 4 decimal places to be the "same"
    pointing_keys = np.round(ra, 4) * 1000 + np.round(dec, 4)
    unique_keys, inv = np.unique(pointing_keys, return_inverse=True)

    n_pointings = len(unique_keys)
    mean_rn  = np.zeros(n_pointings, dtype=np.float64)
    count    = np.zeros(n_pointings, dtype=np.int32)
    ra_pt    = np.zeros(n_pointings, dtype=np.float64) #basically the mean RA across near matches
    dec_pt   = np.zeros(n_pointings, dtype=np.float64) #same here

    np.add.at(mean_rn, inv, rn_sum)
    np.add.at(count,   inv, 1)
    np.add.at(ra_pt,   inv, ra)
    np.add.at(dec_pt,  inv, dec)

    mean_rn /= np.maximum(count, 1)
    ra_pt   /= np.maximum(count, 1)
    dec_pt  /= np.maximum(count, 1)

    print(f"  {len(info)} rows -> {n_pointings} unique pointings.", flush=True)
    print(f"  RA  range: {ra_pt.min():.2f} -- {ra_pt.max():.2f} deg", flush=True)
    print(f"  Dec range: {dec_pt.min():.2f} -- {dec_pt.max():.2f} deg", flush=True)

    return ra_pt, dec_pt, mean_rn


#--------
#OK now we are doing our fun density stuff!
#--------

def grid_and_smooth(ra, dec, values, smooth_deg, grid_resolution=0.1):
    '''
    This function bins and then smoothes the mean values across all pointings.
    Right now I'm binning in cartesian coordinates which is an issue at high Decs...

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
                    increasing from negative (east) to positive (west).
    dec_grid (arr): 1-D array of Dec bin centres in degrees
    smoothed (arr): 2-D array (n_dec, n_lon) of mean of sums of median etc at each pointing,
                    NaN in southern sky
    '''

    #convert RA to longitude
    #increases to the left
    #then we wrap to (-180, 180]
    lon = -(ra % 360.0)
    lon = (lon + 180.0) % 360.0 - 180.0

    lon_min, lon_max = lon.min() - 1, lon.max() + 1
    #shouldn't be any weird data past +90 dec but just to be sure
    dec_min, dec_max = max(dec.min() - 1, -90.0), min(dec.max() + 1, 90.0)

    lon_grid = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
    dec_grid = np.arange(dec_min, dec_max + grid_resolution, grid_resolution)

    n_lon = len(lon_grid)
    n_dec = len(dec_grid)

    col = np.clip(np.round((lon - lon_min) / grid_resolution).astype(int), 0, n_lon - 1)
    row = np.clip(np.round((dec - dec_min) / grid_resolution).astype(int), 0, n_dec - 1)

    grid  = np.full((n_dec, n_lon), np.nan)
    count = np.zeros((n_dec, n_lon))
    accum = np.zeros((n_dec, n_lon))

    for i in range(len(ra)):
        r, c = row[i], col[i]
        accum[r, c] += values[i]
        count[r, c] += 1

    mask = count > 0
    grid[mask] = accum[mask] / count[mask]

    #now we apply a Gaussian filter!
    sigma = smooth_deg / grid_resolution #or should this be divided by number of bins...?
    filled  = np.where(np.isfinite(grid), grid, 0.0)
    weights = np.where(np.isfinite(grid), 1.0,  0.0)
    s_filled  = gaussian_filter(filled,  sigma=sigma)
    s_weights = gaussian_filter(weights, sigma=sigma)
    #there has to be a better way to handle these weird polar cases...
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(s_weights > 1e-6, s_filled / s_weights, np.nan)

    return lon_grid, dec_grid, smoothed

#--------
#argh argh coordinate conversion!
#--------

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


#-------
#plotting stuff
#-------

def _setup_axes():
    '''
    This function sets up the figure and the Mollweide axes!

    Inputs:
    -------
        unmapped_color (str): color of unmapped regions of the sky
    '''

    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor('white')
    ax.grid(True, linestyle=":", alpha=0.6, color="black")

    #my custom xtick marks:
    ax.set_xticklabels([])
    hour_ticks_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    hour_ticks_rad = np.deg2rad(hour_ticks_deg)
    ra_hour_labels = ((-hour_ticks_deg) % 360) / 15.0
    label_dec_rad = np.deg2rad(-15) #can toggle as desired
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
    #                     vmax=np.percentile(finite, 98))
    #else:
    norm = LogNorm(vmin=np.percentile(finite_pos, 2),
                       vmax=np.percentile(finite_pos, 98))

    #convert to radians for pcolormesh... could just save as radians initially
    #maybe easier
    lon_rad = np.deg2rad(lon_grid)
    dec_rad = np.deg2rad(dec_grid) 

    cmap = plt.cm.inferno.copy()
    cmap.set_bad("white")   #could add toggle

    pcm = ax.pcolormesh(lon_rad, dec_rad, smoothed, cmap=cmap, norm=norm,
                        shading="nearest", rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.8, orientation="vertical")
    cbar.set_label('Mean over time, median across DM, and sum across frequency', fontsize=11)

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

    ra_moll  = ra_deg_to_moll_rad(ra)
    dec_moll = dec_deg_to_moll_rad(dec)

    ax.scatter(ra_moll, dec_moll, c="hotpink",
               s=1.5, alpha=0.5, linewidths=0, rasterized=True)

    ax.set_title(f'Pointing Coverage Map', fontsize = 20, fontweight = 'bold')
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

    ra, dec, mean_rn = load_data(npz_path)

    print("Plotting sky map...", flush=True)
    plot_skymap(ra, dec, mean_rn, args.smooth_deg, args.grid_res,
                output_path=args.output_skymap)

    print("Plotting coverage map...", flush=True)
    plot_coverage(ra, dec, output_path=args.output_coverage)


if __name__ == "__main__":
    main()
