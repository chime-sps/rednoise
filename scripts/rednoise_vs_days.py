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
   
    return info[:, :2], median_across_dms[:, :31]
    

def get_dropoff_per_pointing(medians):

    #get decrease scale factor
    medians /= np.nanmin(medians, axis = 1)[:, np.newaxis]
    

    power_dropoff_mean = np.zeros(len(medians))
    power_dropoff_median = np.zeros(len(medians))

    for i in range(len(medians)):
        
        mean_of_medians = np.mean(medians[:(i+1)], axis = 0)
        median_of_medians = np.median(medians[:(i+1)], axis = 0)
        
        power_dropoff_mean[i] = np.sum(1 / mean_of_medians) / len(mean_of_medians)
        power_dropoff_median[i] = np.sum(1 / median_of_medians) / len(median_of_medians)

    
    return power_dropoff_mean, power_dropoff_median

def get_mean_dropoff_per_Ndays(pointings, medians):

    pointing_groups = np.unique(pointings, axis = 0)

    print(f'There are {len(pointing_groups)} unique pointings in our data set.')
    
    mean_per_Ndays = np.zeros(31)
    median_per_Ndays = np.zeros(31)
    count_per_Ndays = np.zeros(31)

    for i in range(len(pointing_groups)):
        
        if i % 100 == 0:
            print(f'{i}/{len(pointing_groups)}')
        pointing_mask = np.all(pointings == pointing_groups[i], axis = 1)

        Ndays = len(pointings[pointing_mask])
        if Ndays > 31:
            Ndays = 31
        
        power_dropoff_mean, power_dropoff_median = get_dropoff_per_pointing(medians[pointing_mask])
        
        if np.all(~np.isnan(power_dropoff_mean)) and np.all(~np.isnan(power_dropoff_median)):

            mean_per_Ndays[:Ndays] += power_dropoff_mean[:Ndays]
            median_per_Ndays[:Ndays] += power_dropoff_median[:Ndays]
            count_per_Ndays[:Ndays] += 1
        
        else:
            print('Ignored one bad pointing.')

    zero_mask = count_per_Ndays == 0.
    mean_per_Ndays = mean_per_Ndays[~zero_mask]
    median_per_Ndays = median_per_Ndays[~zero_mask]
    count_per_Ndays = count_per_Ndays[~zero_mask]

    mean_per_Ndays /= count_per_Ndays
    median_per_Ndays /= count_per_Ndays

    return mean_per_Ndays, median_per_Ndays


def plot_stats(mean_power_dropoff_mean, mean_power_dropoff_median):
    
    Ndays = np.arange(1, len(mean_power_dropoff_mean) + 1)
    fig, ax = plt.subplots(1, figsize = (12, 8))

    ax.plot(Ndays, mean_power_dropoff_mean, 'm', label = 'Mean')
    ax.plot(Ndays, mean_power_dropoff_median, 'c', label = 'Median')
    ax.set_xlabel('Days in Stack')
    ax.set_ylabel('% Power Lost to Rednoise Across Spectrum')
    ax.set_title('Have We Messed Up Our Injections Horribly?')
    plt.legend()
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
    
    pointings, medians = load_data(npz_path, args.max_rows)
    
    dropoff_mean, dropoff_median = get_mean_dropoff_per_Ndays(pointings, medians)

    plot_stats(dropoff_mean, dropoff_median)

if __name__ == '__main__':
    main()
