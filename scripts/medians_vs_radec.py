#!/usr/bin/env python3
'''
This module reads a rednoise info file and produces a figure with two plots;
both show the median rednoise median curve across DMs plotted vs frequency.
One is color-mapped to RA and the other to Dec.

Usage:

    python3 medians_vs_radec.py rednoise_dm_info.npz

To-Do:
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
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection



#-------------
#load our data!
#-------------

def load_data(npz_path: Path, N):
    '''
    This function loads the DM info file and returns the relevant contents.

    Columns: RA, Dec, year, month, day, 0 if wonky RFI in first bins / 1 if not, DM where wonky behavior starts
    
    Inputs:
    -------
        npz_path (Path): path to the DM info file
    
    Returns:
    --------
        RA (arr)     : RA of each median curve
        Dec (arr)    : Dec of each median curve
        medians (arr): median of medians across DM (from info file)
    '''

    print(f"Loading {npz_path}...", flush=True)
    data = np.load(npz_path)

    if N is None:
        N = len(data["info"])
    
    info              = data["info"][:N]               #shape: (N, 7)
    median_across_dms = data["median_across_dms"][:N]  #shape: (N, max_n_freq)

    ra   = info[:, 0]
    dec  = info[:, 1]
    
    N_freq = np.zeros(len(median_across_dms), dtype = int)

    for i in range(len(median_across_dms)):

        zero_mask = median_across_dms[i] == 0.
        N_freq[i] = len(median_across_dms[i, ~zero_mask])

    return ra, dec, N_freq, median_across_dms

def get_scales(N_freq, max_n_freq = 51, b0 = 50, bmax=100000):
    '''
    N_freq (arr): array listing number of windows in each pointing's data

    Note that this is approximate because we don't know how many bins are in the last window.
    We are assuming bmax.
    '''
    print('Calculating scales. I really should add this as a quantity in the h5 file...')

    
    n = np.arange(max_n_freq)
    windows = np.zeros((len(N_freq), max_n_freq))
    windows[:] = np.exp(1 + n / 3) * b0 / np.exp(1)
    too_big = windows > bmax
    windows[too_big] = bmax
    #for i in range(len(N_freq)): 
    #    windows[i, N_freq[i]:] = np.nan
   
    return windows


def freq_to_period(x):
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1 / x, np.inf)

def period_to_freq(x):
    return np.where(x != 0, 1 / x, np.inf)

def plot_medians(ra, dec, medians, scale, N_freq):
    
    print('Generating plot.')
    #fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (18, 6))
    fig, ax = plt.subplots(1, figsize = (14, 8))
    ra_norm = Normalize(vmin = ra.min(), vmax = ra.max())
    ra_colors = plt.cm.BuPu(ra_norm(ra))
    dec_norm = Normalize(vmin = dec.min(), vmax = dec.max())
    dec_colors = plt.cm.rainbow(dec_norm(dec))
    
    freq_idx = np.cumsum(scale, axis = 1).astype(int)
    freq_idx[:, -1] = -1
    all_freqs = np.full(medians.shape, np.nan)
   
    for i in range(len(ra)):
        freq_labels = np.linspace(1e-3, 508, int(np.nansum(scale[i]))+1)
        temp_freq_idx = freq_idx[i, :N_freq[i]]
        temp_freq_idx[-1] = -1
        all_freqs[i, :N_freq[i]] = freq_labels[temp_freq_idx]
    
    segments = [np.column_stack([all_freqs[i], medians[i]]) for i in range(len(all_freqs))]
    #segments = np.column_stack([all_freqs, medians])
   
    #ra_lc = LineCollection(segments, colors=ra_colors, alpha = 0.01, norm = ra_norm)
    #ax[0].add_collection(ra_lc)
    #secax0 = ax[0].secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
    #secax0.set_xlabel('Period (s)')
    #ax[0].set_yscale('log')
    #ax[0].set_xscale('log')
    #ax[0].set_xlabel('Frequency (Hz)')
    #ax[0].set_ylabel('Median Power')
    #cbar = fig.colorbar(ScalarMappable(cmap=plt.cm.BuPu, norm = ra_norm), ax=ax[0])
    #cbar.set_label('RA (degree)')

    dec_lc = LineCollection(segments, colors=dec_colors, alpha = 0.01, norm = dec_norm)
    ax.add_collection(dec_lc)
    secax1 = ax.secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
    secax1.set_xlabel('Period (s)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Median Power')
    ax.set_title('Rednoise Power Across Declination')
    cbar = fig.colorbar(ScalarMappable(cmap=plt.cm.rainbow, norm = dec_norm), ax=ax)
    cbar.set_label('Dec (degree)')
    
    plt.savefig('Rednoise_vs_Dec.png')
    plt.show()

def main():

    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_file",
                        help="Path to rednoise_dm_info.npz")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Only process first N rows.")

    args = parser.parse_args()

    npz_path = Path(args.npz_file)
    
    if not npz_path.exists():
        sys.exit(f"File not found: {npz_path}")
    
    ra, dec, N_freq, median_across_dms = load_data(npz_path, args.max_rows)
    
    scales = get_scales(N_freq)

    plot_medians(ra, dec, median_across_dms, scales, N_freq)

if __name__ == '__main__':
    main()
