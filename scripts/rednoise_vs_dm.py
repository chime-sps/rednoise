#!/usr/bin/env python3
'''
rednoise_vs_dm.py
=================
Reads the combined rednoise medians HDF5 file (the output of
combine_rednoise_medians.py) and plots, for a selection of pointings
(by default Dec 90 +/- 5 deg), the rednoise medians as a function of
frequency -- one line per DM trial, colored by trial DM on a jet
colormap at low alpha.

Selecting pointings
-------------------
    (default)                   Dec within 5 deg of 90, i.e. Dec >= 85
    --dec-center C --dec-radius R
                                C - R <= Dec <= C + R  (e.g. --dec-center 40
                                --dec-radius 2 -> Dec 38-42). Either one alone
                                uses the default for the other (90, 5).
    --zenith                    shorthand for --dec-center CHIME_LAT (49.32):
                                zenith distance at transit <= --dec-radius
    --b-center C --b-radius R   C - R <= b <= C + R, b = galactic latitude
                                (from each row's RA/Dec, J2000). Either one
                                alone uses the default for the other (0, 5).
                                On its own, this replaces the default Dec cut;
                                combined with --dec-center/--dec-radius/--zenith,
                                rows must pass both cuts.

What gets averaged
------------------
1. Rows (pointing-days) passing the selection above are kept.
2. Rows are grouped into pointings by matching each row's RA/Dec to the
   nearest beam in pointings_map_v2-0.json (for every row, including ones
   from before the v1-3 -> v2-0 cutover, so a beam whose map coordinates
   drifted slightly between map versions is still counted as one pointing).
3. For each pointing, the MEAN over its days is taken for every
   (DM index, frequency bin) cell, over each row's valid region
   medians[i, :n_dm[i], :n_freq[i]] only, skipping non-finite values.
4. The MEDIAN over pointings of those per-pointing means is then taken for
   every cell (a pointing only contributes to DM trials it actually
   searched). The result has shape (n_DM, n_freq_bins).

Frequency axis
--------------
The "scale" dataset says how many raw power-spectrum bins went into each
rednoise bin. Raw bins are linearly spaced from 0 up to the Nyquist
frequency f_nyq = 1 / (2 * TSAMP), so for a row with n_raw = sum(scale) raw
bins the spacing is df = f_nyq / (n_raw - 1), and each rednoise bin gets the
mean frequency of the raw bins it covers. n_raw depends on the observation
length, so the plotted frequency of each bin is the mean over all selected
rows with a valid scale.

DM axis
-------
The combined medians file stores DM trials by index only. The maximum DM
searched at each row's pointing is looked up in the pointing maps in
scripts/data (v1-3 for rows before POINTINGS_MAP_CUTOVER, v2-0 on/after --
same convention and nearest-neighbor matching as rednoise_skymap.py), capped
at the pipeline's dedisp.maxdm config value (--maxdm-cap, default 1700, as in
sps_pipeline's sps_config.yml). The FDMT dedispersion trials are linearly
spaced from 0 to that maximum DM, so the trial DMs of a row with n_dm trials
are linspace(0, maxdm, n_dm). The trial DM plotted for each index is the mean
of that over all selected rows matched to the map.

Usage
-----
    python3 rednoise_vs_dm.py combined_medians.h5
    python3 rednoise_vs_dm.py combined_medians.h5 --dec-center 40 --dec-radius 2 \\
        --alpha 0.05 --output-plot rednoise_vs_dm.png --output-npz rednoise_vs_dm.npz
    python3 rednoise_vs_dm.py combined_medians.h5 --zenith
    python3 rednoise_vs_dm.py combined_medians.h5 --b-center 0 --b-radius 5

    # replot from a saved --output-npz file without re-reading the h5 file
    python3 rednoise_vs_dm.py rednoise_vs_dm.npz --output-plot replot.png --ylim 1e9 1e13
    python3 rednoise_vs_dm.py rednoise_vs_dm.npz --title "Polar cap, Sept 2026"

Replotting from a saved .npz
----------------------------
--output-npz saves the plotted arrays (median, freqs, dms, counts) together
with the selection that produced them. Passing that .npz instead of the h5
file redraws the plot in seconds, with any --xlim/--ylim/--dm-lim/--alpha/
--dpi/--title; plot_from_npz() does the same from Python. Selection options can't be
used with an .npz input (its data is already selected).

Plot limits
-----------
The frequency axis, power axis and DM colorbar use the same fixed limits on
every run (DEFAULT_XLIM, DEFAULT_YLIM, DEFAULT_DM_LIM), so plots of different
selections line up. Change them with --xlim, --ylim and --dm-lim; a warning
is printed if any data fall outside them.

--colorbar-log colors lines by log(DM) instead of DM, which spreads out the
low-DM trials. A log scale can't include DM 0, so its default colorbar range
is DEFAULT_DM_LIM_LOG (1-1600). Trials below 1 (including DM 0) are drawn in
the lowest color, and trials above 1600 in the highest.
'''

import json
import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import click
import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# my obsessive matplotlib formatting:
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'monospace'
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

import sps_common.constants as _sps_constants
from sps_common.constants import TSAMP

# CHIME's latitude = declination of the zenith. Fall back to the value in
# sps_common/constants.py for sps_common versions without CHIME_LAT.
CHIME_LAT = getattr(_sps_constants, "CHIME_LAT", 49.3211)

# pointing-map conventions shared with the skymap script
from rednoise_skymap import (
    MAX_MATCH_SEP_DEG,
    POINTINGS_MAP_CUTOVER,
    POINTINGS_MAP_V1_3_PATH,
    POINTINGS_MAP_V2_0_PATH,
    _radec_to_xyz,
    nearest_pointing_values,
)


DEFAULT_DEC_CENTER = 90.0
DEFAULT_DEC_RADIUS = 5.0
DEFAULT_B_CENTER = 0.0
DEFAULT_B_RADIUS = 5.0

# Rotation from ICRS (~J2000 equatorial) unit vectors to Galactic unit
# vectors (Hipparcos / astropy definition of the Galactic frame).
_ICRS_TO_GALACTIC = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])
DEFAULT_MAXDM_CAP = 1700.0   # dedisp.maxdm in sps_pipeline/sps_config.yml
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)   # chunk-decompressing processes

# Fixed plot limits, so plots of different selections can be compared
# directly. Override with --xlim/--ylim/--dm-lim.
#   x: covers the lowest rednoise bin (~0.02 Hz for a 2^20-sample spectrum)
#      up to the Nyquist frequency 1 / (2 * TSAMP) ~ 509 Hz
#   y: rednoise medians are ~1e9-1e11 per bin away from the low-frequency
#      rise (the skymap's sum over ~45 bins is ~5e10-5e12), higher below
#   DM: 0 up to the pipeline's dedisp.maxdm cap, the largest trial DM searched
DEFAULT_XLIM = (1e-3, 1e3)
DEFAULT_YLIM = (1e8, 1e14)
DEFAULT_DM_LIM = (0.0, DEFAULT_MAXDM_CAP)
#   log DM: 1 to 1600 pc cm^-3 (a log scale can't reach 0); trials below 1,
#   including DM 0, get the lowest color and trials above 1600 the highest
DEFAULT_DM_LIM_LOG = (1.0, 1600.0)


# -------------
# selection + frequency helpers
# -------------

def galactic_latitude(ra, dec):
    '''
    Galactic latitude b (degrees) of equatorial (RA, Dec) in degrees, J2000.
    '''
    ra = np.deg2rad(np.asarray(ra, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec, dtype=np.float64))
    xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    z_gal = np.tensordot(_ICRS_TO_GALACTIC[2], xyz, axes=1)
    return np.rad2deg(np.arcsin(np.clip(z_gal, -1.0, 1.0)))


@dataclass(frozen=True)
class Selection:
    '''
    Which rows to use. `dec` and `b` are (center, radius) in degrees, or
    None for no cut on that coordinate; a row must pass every cut given.
    `zenith` only changes the label (the dec center is already CHIME_LAT).
    '''
    dec: tuple = (DEFAULT_DEC_CENTER, DEFAULT_DEC_RADIUS)
    b: tuple = None
    zenith: bool = False

    def mask(self, ra, dec):
        ra = np.asarray(ra, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)
        keep = np.isfinite(ra) & np.isfinite(dec)
        if self.dec is not None:
            c, r = self.dec
            keep &= np.abs(dec - c) <= r
        if self.b is not None:
            c, r = self.b
            with np.errstate(invalid="ignore"):
                keep &= np.abs(galactic_latitude(ra, dec) - c) <= r
        return keep

    @property
    def label(self):
        parts = []
        if self.dec is not None:
            c, r = self.dec
            parts.append(f"≤ {r:g}° from zenith" if self.zenith else f"Dec {c:g}° ± {r:g}°")
        if self.b is not None:
            c, r = self.b
            parts.append(f"b {c:g}° ± {r:g}°")
        return ", ".join(parts) if parts else "all pointings"


def select_rows(ra, dec, selection=None):
    '''
    Return the (sorted) row indices passing `selection` (default: Selection()).
    '''
    selection = Selection() if selection is None else selection
    return np.flatnonzero(selection.mask(ra, dec))


def resolve_selection(zenith=False, dec_center=None, dec_radius=None,
                      b_center=None, b_radius=None):
    '''
    Turn the command-line options into a Selection.

      - no options                  -> Dec 90 +/- 5
      - --dec-center/--dec-radius   -> Dec cut (missing one takes its default)
      - --zenith                    -> Dec cut centered on CHIME_LAT
      - --b-center/--b-radius       -> galactic-latitude cut (missing one takes
                                       its default); replaces the default Dec
                                       cut unless a Dec option is also given,
                                       in which case both cuts apply
    '''
    if zenith and dec_center is not None:
        raise ValueError("--zenith can't be combined with --dec-center "
                         "(it sets the center to CHIME_LAT).")
    for name, r in (("--dec-radius", dec_radius), ("--b-radius", b_radius)):
        if r is not None and r < 0:
            raise ValueError(f"{name} must be >= 0.")

    dec_given = zenith or dec_center is not None or dec_radius is not None
    b_given = b_center is not None or b_radius is not None

    dec_cut = None
    if dec_given or not b_given:
        center = CHIME_LAT if zenith else (DEFAULT_DEC_CENTER if dec_center is None else dec_center)
        radius = DEFAULT_DEC_RADIUS if dec_radius is None else dec_radius
        dec_cut = (float(center), float(radius))

    b_cut = None
    if b_given:
        b_cut = (float(DEFAULT_B_CENTER if b_center is None else b_center),
                 float(DEFAULT_B_RADIUS if b_radius is None else b_radius))

    return Selection(dec=dec_cut, b=b_cut, zenith=bool(zenith))


def scale_to_frequencies(scale, tsamp=TSAMP):
    '''
    Convert one row's (unpadded) scale array into the frequency of each
    rednoise bin.

    The raw power-spectrum bins are linearly spaced from 0 to the Nyquist
    frequency 1 / (2 * tsamp), n_raw = sum(scale) of them in total. Rednoise
    bin j covers raw bins [c_{j-1}, c_j) where c = cumsum(scale), and is
    assigned the mean frequency of those raw bins.

    Inputs:
    -------
        scale (arr): number of raw bins in each rednoise bin, shape (n_freq,)
        tsamp (float): sampling time, seconds

    Returns:
    --------
        freqs (arr): frequency of each rednoise bin in Hz, shape (n_freq,),
                     or None if scale is empty or has non-positive entries
                     (e.g. the -1 "missing" sentinel)
    '''
    scale = np.rint(np.asarray(scale, dtype=np.float64)).astype(np.int64)
    if scale.size == 0 or np.any(scale <= 0):
        return None

    n_raw = int(scale.sum())
    f_nyq = 1.0 / (2.0 * tsamp)
    df = f_nyq / max(n_raw - 1, 1)

    end = np.cumsum(scale)          # one past the last raw bin in each window
    start = end - scale             # first raw bin in each window
    # mean of k * df for k = start .. end-1
    return 0.5 * (start + end - 1) * df


# -------------
# pointing maps: max DM + pointing identity
# -------------

def load_pointing_map(pointings_map_path):
    '''
    Load a pointings_map_*.json and build a nearest-neighbor tree over it.

    Returns:
    --------
        tree (cKDTree): over the map's (RA, Dec), 3D unit-vector space
        maxdm (arr): "maxdm" field, same order as the tree's points
    '''
    print(f"Loading pointing map {pointings_map_path} ...", flush=True)
    with open(pointings_map_path, "r") as f:
        pointings = json.load(f)
    ra = np.array([p["ra"] for p in pointings], dtype=np.float64)
    dec = np.array([p["dec"] for p in pointings], dtype=np.float64)
    maxdm = np.array([p["maxdm"] for p in pointings], dtype=np.float64)
    return cKDTree(_radec_to_xyz(ra, dec)), maxdm


def lookup_pointings(ra, dec, year, month, day,
                     map_v1_3, map_v2_0, maxdm_cap=DEFAULT_MAXDM_CAP):
    '''
    For each row, look up the max DM searched and a pointing ID.

    Inputs:
    -------
        ra, dec (arr): row coordinates, degrees
        year, month, day (arr): row dates
        map_v1_3, map_v2_0 (tuple): (tree, maxdm) from load_pointing_map()
        maxdm_cap (float or None): pipeline dedisp.maxdm cap

    Returns:
    --------
        maxdm (arr): max DM searched, min(map maxdm, maxdm_cap); from v1-3
                     before POINTINGS_MAP_CUTOVER, v2-0 on/after; NaN if the
                     row doesn't match any beam within MAX_MATCH_SEP_DEG
        pointing_id (arr): int; the index of the nearest v2-0 beam, or a
                     negative ID (one per distinct RA/Dec) for rows with no
                     v2-0 match
    '''
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    row_date = (np.asarray(year, dtype=np.int64) * 10000
                + np.asarray(month, dtype=np.int64) * 100
                + np.asarray(day, dtype=np.int64))

    tree_v1, maxdm_v1 = map_v1_3
    tree_v2, maxdm_v2 = map_v2_0

    maxdm = np.where(
        row_date < POINTINGS_MAP_CUTOVER,
        nearest_pointing_values(tree_v1, maxdm_v1, ra, dec),
        nearest_pointing_values(tree_v2, maxdm_v2, ra, dec),
    )
    if maxdm_cap is not None:
        maxdm = np.minimum(maxdm, maxdm_cap)   # NaN stays NaN

    idx = nearest_pointing_values(tree_v2, np.arange(len(maxdm_v2), dtype=np.float64),
                                  ra, dec)
    pointing_id = np.where(np.isnan(idx), -1, idx).astype(np.int64)
    unmatched = pointing_id < 0
    if np.any(unmatched):
        _, inv = np.unique(np.column_stack([np.round(ra[unmatched], 4),
                                            np.round(dec[unmatched], 4)]),
                           axis=0, return_inverse=True)
        pointing_id[unmatched] = -1 - np.ravel(inv)

    return maxdm, pointing_id


def trial_dms(maxdm, n_dm):
    '''
    Trial DMs of a row: FDMT trials are linearly spaced from 0 to maxdm.
    '''
    return np.linspace(0.0, maxdm, int(n_dm))


# -------------
# reading medians: one pass over the HDF5 chunks
# -------------
#
# combine_rednoise_medians.py stores "medians" gzip-compressed in chunks of
# (256 rows, max_n_dm, max_n_freq) -- ~877 MB uncompressed each at
# (256, 16799, 51). HDF5 can't decompress part of a chunk, so reading even
# one row costs a full chunk decompression. The rows for one pointing are
# spread across many chunks (one per day), and each chunk holds rows from
# many pointings, so reading pointing-by-pointing decompresses the same
# chunk over and over. Instead, the selected rows are read in file order,
# one read per chunk (so every chunk is decompressed exactly once), in
# parallel worker processes (h5py serializes all HDF5 calls within one
# process, so threads wouldn't help), and accumulated per pointing.

def _row_count(h5):
    '''
    Use the minimum length across the datasets we need, in case a killed
    combine run left them one batch out of sync (see rednoise_dm_behavior.py).
    '''
    names = ["medians", "scale", "n_dm", "n_freq", "ra", "dec", "year", "month", "day"]
    return min(h5[name].shape[0] for name in names)


def _read_medians(dataset, rows, max_dm, max_freq):
    '''
    Read medians[rows, :max_dm, :max_freq] (rows sorted, unique). Falls back
    to row-by-row reads if a corrupted HDF5 chunk makes the bulk read fail,
    skipping only the bad rows. Returns (array, rows_actually_read).
    '''
    try:
        return dataset[list(rows), :max_dm, :max_freq], np.asarray(rows)
    except OSError as e:
        print(f"  [WARN] Chunk read failed ({e}); falling back to row-by-row reads.",
              flush=True)

    out, good = [], []
    for r in rows:
        try:
            out.append(dataset[int(r), :max_dm, :max_freq])
            good.append(r)
        except OSError:
            print(f"    [SKIP] Corrupted row {r}", flush=True)
    if not out:
        return np.empty((0, max_dm, max_freq), dtype=np.float32), np.asarray(good, dtype=int)
    return np.stack(out), np.asarray(good)


def batch_rows_by_chunk(rows, rows_per_chunk):
    '''
    Split sorted row indices into batches that each fall in one HDF5 chunk
    along the row axis, so each batch is one read and one decompression.
    '''
    rows = np.asarray(rows)
    if len(rows) == 0:
        return []
    keys = rows // max(int(rows_per_chunk), 1)
    return np.split(rows, np.flatnonzero(np.diff(keys)) + 1)


# Per-process state for the reader workers (each opens its own file handle).
_READER = {}


def _init_reader(h5_path, max_dm, max_freq):
    _READER["file"] = h5py.File(h5_path, "r")
    _READER["ds"] = _READER["file"]["medians"]
    _READER["shape"] = (max_dm, max_freq)


def _read_batch(rows):
    return _read_medians(_READER["ds"], rows, *_READER["shape"])


def _close_reader():
    f = _READER.pop("file", None)
    if f is not None:
        f.close()
    _READER.clear()


def _iter_batches(h5_path, batches, max_dm, max_freq, workers):
    '''
    Yield (medians, rows_read) for every batch, reading in `workers`
    processes (in-process if workers <= 1). Order is not preserved.
    '''
    if workers <= 1 or len(batches) <= 1:
        _init_reader(h5_path, max_dm, max_freq)
        try:
            for b in batches:
                yield _read_batch(b)
        finally:
            _close_reader()
        return

    methods = mp.get_all_start_methods()
    ctx = mp.get_context("fork" if "fork" in methods else None)
    with ctx.Pool(workers, initializer=_init_reader,
                  initargs=(str(h5_path), max_dm, max_freq)) as pool:
        yield from pool.imap_unordered(_read_batch, batches)


# -------------
# the actual averaging
# -------------

def rednoise_vs_dm(h5_path, map_v1_3, map_v2_0, selection=None,
                   maxdm_cap=DEFAULT_MAXDM_CAP, workers=DEFAULT_WORKERS, tsamp=TSAMP):
    '''
    Median over pointings of the mean over days of the rednoise medians,
    for the rows passing `selection` (default: Dec 90 +/- 5).

    Inputs:
    -------
        h5_path (Path): combined medians file
        map_v1_3, map_v2_0 (tuple): from load_pointing_map()
        selection (Selection): which rows to use; see resolve_selection()
        maxdm_cap (float or None): pipeline dedisp.maxdm cap
        workers (int): processes decompressing HDF5 chunks in parallel
        tsamp (float): sampling time, seconds

    Returns (dict):
    --------
        median (arr): (n_dm, n_freq); NaN where no pointing has data
        freqs (arr): (n_freq,) frequency of each bin, Hz
        dms (arr): (n_dm,) trial DM of each index, pc cm^-3; NaN if no
                   map-matched row has that trial
        n_pointings_per_cell (arr): (n_dm, n_freq) pointings in each median
        n_pointings (int), n_rows (int): pointings / rows used
    '''
    selection = Selection() if selection is None else selection

    # ---- metadata: small, read in full up front; file closed before forking
    with h5py.File(h5_path, "r") as h5:
        N = _row_count(h5)
        ra = h5["ra"][:N]
        dec = h5["dec"][:N]
        year, month, day = h5["year"][:N], h5["month"][:N], h5["day"][:N]
        n_dm_all = h5["n_dm"][:N].astype(int)
        n_freq_all = h5["n_freq"][:N].astype(int)

        rows = select_rows(ra, dec, selection)
        rows = rows[(n_dm_all[rows] > 0) & (n_freq_all[rows] > 0)]
        if len(rows) == 0:
            raise ValueError(f"No rows with {selection.label} in this file.")

        max_dm = int(n_dm_all[rows].max())
        max_freq = int(n_freq_all[rows].max())
        scales = h5["scale"][list(rows), :max_freq]

        ds = h5["medians"]
        rows_per_chunk = ds.chunks[0] if ds.chunks else 64
        chunk_mb = (np.prod(ds.chunks) * ds.dtype.itemsize / 1e6) if ds.chunks else None
        n_chunks_total = int(np.ceil(N / rows_per_chunk))

    maxdm, pid = lookup_pointings(ra[rows], dec[rows], year[rows], month[rows],
                                  day[rows], map_v1_3, map_v2_0, maxdm_cap)
    groups, group_of_row = np.unique(pid, return_inverse=True)
    group_of_row = np.ravel(group_of_row)
    n_groups = len(groups)
    print(f"{len(rows)}/{N} rows with {selection.label}, "
          f"from {n_groups} pointing(s).", flush=True)

    n_unmatched = int(np.sum(np.isnan(maxdm)))
    if n_unmatched:
        print(f"  [WARN] {n_unmatched} row(s) matched no pointing-map beam within "
              f"{MAX_MATCH_SEP_DEG} deg; they're used for the medians but not "
              f"the DM axis.", flush=True)
    if n_unmatched == len(rows):
        raise ValueError("No selected row matched the pointing maps; can't label trial DMs.")

    # ---- frequency axis (from scale) and DM axis (from pointing maps): no medians needed
    freq_total = np.zeros(max_freq, dtype=np.float64)
    freq_counts = np.zeros(max_freq, dtype=np.int64)
    freq_min = np.full(max_freq, np.inf)
    freq_max = np.full(max_freq, -np.inf)
    dm_total = np.zeros(max_dm, dtype=np.float64)
    dm_counts = np.zeros(max_dm, dtype=np.int64)
    dm_steps = []
    n_bad_scale = 0
    for pos, r in enumerate(rows):
        nd, nf = int(n_dm_all[r]), int(n_freq_all[r])
        f = scale_to_frequencies(scales[pos, :nf], tsamp=tsamp)
        if f is None:
            n_bad_scale += 1
        else:
            freq_total[:nf] += f
            freq_counts[:nf] += 1
            freq_min[:nf] = np.minimum(freq_min[:nf], f)
            freq_max[:nf] = np.maximum(freq_max[:nf], f)
        if not np.isnan(maxdm[pos]):
            dm_total[:nd] += trial_dms(maxdm[pos], nd)
            dm_counts[:nd] += 1
            if nd > 1:
                dm_steps.append(maxdm[pos] / (nd - 1))

    if n_bad_scale:
        print(f"  [WARN] {n_bad_scale} row(s) had a missing/invalid scale; their "
              f"medians are used but they don't contribute to the frequency axis.",
              flush=True)
    if not np.any(freq_counts):
        raise ValueError("No selected row has a valid 'scale'; cannot build a frequency axis.")

    # ---- per-pointing sums over days, reading every HDF5 chunk once
    batches = batch_rows_by_chunk(rows, rows_per_chunk)
    size = f" (~{chunk_mb:.0f} MB uncompressed each)" if chunk_mb else ""
    print(f"  Selected rows live in {len(batches)}/{n_chunks_total} HDF5 chunks{size}; "
          f"decompressing each once with {workers} worker(s).", flush=True)

    acc_gb = n_groups * max_dm * max_freq * (8 + 2) / 1e9
    print(f"  Per-pointing accumulators: ({n_groups}, {max_dm}, {max_freq}), "
          f"~{acc_gb:.2f} GB", flush=True)
    total = np.zeros((n_groups, max_dm, max_freq), dtype=np.float64)
    counts = np.zeros((n_groups, max_dm, max_freq), dtype=np.uint16)

    with tqdm(total=len(batches), desc="Reading chunks", unit="chunk",
              dynamic_ncols=True, smoothing=0.05) as bar:
        for medians, good in _iter_batches(h5_path, batches, max_dm, max_freq, workers):
            for m, r in zip(medians, good):
                pos = int(np.searchsorted(rows, r))
                g = group_of_row[pos]
                nd, nf = int(n_dm_all[r]), int(n_freq_all[r])
                valid = m[:nd, :nf].astype(np.float64)
                ok = np.isfinite(valid)
                total[g, :nd, :nf] += np.where(ok, valid, 0.0)
                counts[g, :nd, :nf] += ok
            bar.update(1)

    with np.errstate(invalid="ignore", divide="ignore"):
        per_pointing = np.where(counts > 0, total / counts, np.nan).astype(np.float32)
    del total

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cells
        median = np.nanmedian(per_pointing, axis=0)
    n_pointings_per_cell = np.sum(np.isfinite(per_pointing), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        freqs = np.where(freq_counts > 0, freq_total / freq_counts, np.nan)
        spread = np.where(freq_counts > 0, (freq_max - freq_min) / freqs, 0.0)
        dms = np.where(dm_counts > 0, dm_total / dm_counts, np.nan)

    if np.nanmax(spread) > 0.01:
        print(f"  [INFO] Bin frequencies differ by up to {100 * np.nanmax(spread):.1f}% "
              f"between rows (different observation lengths); plotting their mean.",
              flush=True)
    if dm_steps:
        dm_steps = np.asarray(dm_steps)
        print(f"  Implied DM step: median {np.median(dm_steps):.4f} pc/cm^3 "
              f"(range {dm_steps.min():.4f}-{dm_steps.max():.4f})", flush=True)

    return dict(median=median, freqs=freqs, dms=dms,
                n_pointings_per_cell=n_pointings_per_cell,
                n_pointings=n_groups, n_rows=len(rows))

# -------------
# plotting
# -------------

def freq_to_period(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1 / x, np.inf)


def period_to_freq(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1 / x, np.inf)


def build_segments(curves, freqs):
    '''
    One (freq, power) polyline per DM row, keeping only points that are
    finite and positive (so they can sit on log axes).
    '''
    segments = []
    for row in curves:
        ok = np.isfinite(freqs) & np.isfinite(row) & (freqs > 0) & (row > 0)
        segments.append(np.column_stack([freqs[ok], row[ok]]))
    return segments


def _check_limits(name, lo, hi, log=False):
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"{name} must be two finite numbers with min < max, got ({lo}, {hi}).")
    if log and lo <= 0:
        raise ValueError(f"{name} is on a log axis, so its min must be > 0, got {lo}.")


def _warn_outside(name, values, lo, hi, unit=""):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n_out = int(np.sum((values < lo) | (values > hi)))
    if n_out:
        print(f"  [WARN] {n_out}/{len(values)} {name} value(s) fall outside the fixed "
              f"limits [{lo:g}, {hi:g}]{unit} (data span {values.min():.3g} to "
              f"{values.max():.3g}); adjust with --{name.split()[0]}.", flush=True)
    return n_out


def plot_rednoise_vs_dm(curves, freqs, dms, alpha=0.05, n_pointings=None,
                        dpi=150, output_path=None, selection_label=None,
                        xlim=DEFAULT_XLIM, ylim=DEFAULT_YLIM, dm_lim=None,
                        title=None, colorbar_log=False):
    '''
    Plot each DM trial's rednoise curve vs frequency, colored by trial DM.
    DM rows with a non-finite DM label are not drawn.

    Inputs:
    -------
        curves (arr): (n_dm, n_freq) from rednoise_vs_dm()["median"]
        freqs (arr): (n_freq,) Hz
        dms (arr): (n_dm,) trial DM of each row, pc cm^-3
        alpha (float): line transparency
        output_path (Path): save here if given, otherwise plt.show()
        selection_label (str): title text for the pointing selection;
                               default Selection().label
        xlim, ylim (tuple): fixed frequency (Hz) / power axis limits
        dm_lim (tuple): fixed colorbar limits, pc cm^-3; lines outside get
                        the end colors of the colormap. Default DEFAULT_DM_LIM,
                        or DEFAULT_DM_LIM_LOG with colorbar_log
        title (str): plot title, used exactly as given (matplotlib mathtext
                     like '$b$' works; "" for no title). Default: built from
                     selection_label and n_pointings.
        colorbar_log (bool): color by log(DM) instead of DM; dm_lim[0] must
                     then be > 0; DMs below it (e.g. DM 0) get the lowest color
                     without a warning, DMs above it the highest (with a warning)

    Returns:
    --------
        fig, ax, line_collection
    '''
    print("Generating plot.", flush=True)
    if dm_lim is None:
        dm_lim = DEFAULT_DM_LIM_LOG if colorbar_log else DEFAULT_DM_LIM
    _check_limits("xlim", *xlim, log=True)
    _check_limits("ylim", *ylim, log=True)
    _check_limits("dm-lim", *dm_lim, log=colorbar_log)
    fig, ax = plt.subplots(1, figsize=(14, 8), dpi=dpi)

    dms = np.asarray(dms, dtype=np.float64)
    has_dm = np.isfinite(dms)
    curves, dms = np.asarray(curves)[has_dm], dms[has_dm]

    # DMs beyond dm_lim get the end colors: clip them first, since LogNorm
    # would otherwise turn DM 0 into a masked (invisible) value
    norm_cls = LogNorm if colorbar_log else Normalize
    norm = norm_cls(vmin=dm_lim[0], vmax=dm_lim[1], clip=True)
    colors = plt.cm.jet(norm(np.clip(dms, dm_lim[0], dm_lim[1])))

    segments = build_segments(curves, freqs)
    lc = LineCollection(segments, colors=colors, alpha=alpha)
    ax.add_collection(lc)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    pts = [s for s in segments if len(s)]
    if pts:
        pts = np.concatenate(pts)
        _warn_outside("xlim (frequency)", pts[:, 0], *xlim, " Hz")
        _warn_outside("ylim (power)", pts[:, 1], *ylim)
    # with a log colorbar, low trials (DM 0 up to the minimum) are expected to
    # sit below the range -- only warn about DMs above it
    _warn_outside("dm-lim (trial DM)", dms[dms >= dm_lim[0]] if colorbar_log else dms,
                  *dm_lim, " pc cm^-3")

    secax = ax.secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
    secax.set_xlabel('Period (s)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$\mathrm{median}_{\mathrm{pointings}}\ \langle P_{\mathrm{rednoise}} \rangle_{\mathrm{days}}$')
    if selection_label is None:
        selection_label = Selection().label
    if title is None:
        title = f'Rednoise vs DM ({selection_label}'
        title += f', {n_pointings} pointings)' if n_pointings is not None else ')'
    ax.set_title(title, pad=12)

    cbar = fig.colorbar(ScalarMappable(cmap=plt.cm.jet, norm=norm), ax=ax)
    cbar.set_label(r'Trial DM (pc cm$^{-3}$)')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to {output_path}", flush=True)
    else:
        plt.show()

    return fig, ax, lc


# -------------
# saving + replotting from .npz
# -------------

NPZ_REQUIRED_KEYS = ("median", "freqs", "dms")


def save_results(npz_path, res, selection):
    '''
    Save rednoise_vs_dm() output plus the selection that produced it, so
    plot_from_npz() can redraw it with the right title.
    '''
    extra = dict(selection_label=np.array(selection.label))
    if selection.dec is not None:
        extra["selection_dec"] = np.array(selection.dec, dtype=np.float64)
    if selection.b is not None:
        extra["selection_b"] = np.array(selection.b, dtype=np.float64)
    np.savez_compressed(npz_path, **res, **extra)


def load_results(npz_path):
    '''
    Load a .npz written by save_results() (or by --output-npz).

    Returns (dict):
    --------
        median (arr): (n_dm, n_freq)
        freqs (arr): (n_freq,)
        dms (arr): (n_dm,)
        n_pointings (int or None)
        selection_label (str): what was selected; for .npz files saved
                               before the label was stored, the file name
    '''
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [k for k in NPZ_REQUIRED_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{npz_path} is missing {missing}; expected a file saved "
                             f"by rednoise_vs_dm.py --output-npz.")
        median = np.asarray(data["median"], dtype=np.float64)
        freqs = np.asarray(data["freqs"], dtype=np.float64)
        dms = np.asarray(data["dms"], dtype=np.float64)
        n_pointings = int(data["n_pointings"]) if "n_pointings" in data.files else None
        label = str(data["selection_label"]) if "selection_label" in data.files else npz_path.name

    if median.ndim != 2 or median.shape != (len(dms), len(freqs)):
        raise ValueError(f"{npz_path}: median has shape {median.shape}, expected "
                         f"(len(dms), len(freqs)) = ({len(dms)}, {len(freqs)}).")
    return dict(median=median, freqs=freqs, dms=dms, n_pointings=n_pointings,
                selection_label=label)


def plot_from_npz(npz_path, output_path=None, alpha=0.05, dpi=150,
                  xlim=DEFAULT_XLIM, ylim=DEFAULT_YLIM, dm_lim=None,
                  title=None, colorbar_log=False):
    '''
    Redraw the rednoise-vs-DM plot from a .npz saved with --output-npz,
    without re-reading the combined medians file.

    Inputs:
    -------
        npz_path (Path): file saved by --output-npz / save_results()
        output_path (Path): save the plot here, otherwise plt.show()
        alpha, dpi, xlim, ylim, dm_lim, title, colorbar_log: as in
                  plot_rednoise_vs_dm()
                  (default title: built from the selection saved in the .npz)

    Returns:
    --------
        fig, ax, line_collection
    '''
    print(f"Loading {npz_path} ...", flush=True)
    res = load_results(npz_path)
    return plot_rednoise_vs_dm(res["median"], res["freqs"], res["dms"], alpha=alpha,
                               n_pointings=res["n_pointings"], dpi=dpi,
                               output_path=output_path,
                               selection_label=res["selection_label"],
                               xlim=xlim, ylim=ylim, dm_lim=dm_lim, title=title,
                               colorbar_log=colorbar_log)


def _is_npz(path):
    return Path(path).suffix.lower() == ".npz"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dec-center", type=float, default=None,
              help=f"Use pointings with |Dec - DEC_CENTER| <= DEC_RADIUS, degrees "
                   f"[default: {DEFAULT_DEC_CENTER:g}].")
@click.option("--dec-radius", type=float, default=None,
              help=f"Declination radius, degrees [default: {DEFAULT_DEC_RADIUS:g}].")
@click.option("--zenith", is_flag=True, default=False,
              help=f"Center the Dec cut on zenith (Dec = CHIME_LAT = {CHIME_LAT:g}); "
                   f"radius from --dec-radius. Can't be used with --dec-center.")
@click.option("--b-center", type=float, default=None,
              help=f"Use pointings with |b - B_CENTER| <= B_RADIUS, b = galactic "
                   f"latitude, degrees [default: {DEFAULT_B_CENTER:g}]. Replaces the "
                   f"default Dec cut; with a Dec option too, both cuts apply.")
@click.option("--b-radius", type=float, default=None,
              help=f"Galactic latitude radius, degrees [default: {DEFAULT_B_RADIUS:g}].")
@click.option("--maxdm-cap", type=float, default=DEFAULT_MAXDM_CAP, show_default=True,
              help="Pipeline dedisp.maxdm cap applied on top of the pointing map's maxdm.")
@click.option("--pointings-map-v1-3", "map_v1_3_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=POINTINGS_MAP_V1_3_PATH, show_default=True,
              help="Pointing map used for rows before the v2-0 cutover.")
@click.option("--pointings-map-v2-0", "map_v2_0_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=POINTINGS_MAP_V2_0_PATH, show_default=True,
              help="Pointing map used on/after the cutover, and for pointing IDs.")
@click.option("--xlim", type=(float, float), default=DEFAULT_XLIM, show_default=True,
              metavar="MIN MAX", help="Fixed frequency axis limits, Hz.")
@click.option("--ylim", type=(float, float), default=DEFAULT_YLIM, show_default=True,
              metavar="MIN MAX", help="Fixed power axis limits.")
@click.option("--dm-lim", type=(float, float), default=None, metavar="MIN MAX",
              help=f"Fixed DM colorbar limits, pc cm^-3 [default: "
                   f"{DEFAULT_DM_LIM[0]:g} {DEFAULT_DM_LIM[1]:g}, or "
                   f"{DEFAULT_DM_LIM_LOG[0]:g} {DEFAULT_DM_LIM_LOG[1]:g} with --colorbar-log].")
@click.option("--colorbar-log", is_flag=True, default=False,
              help="Color lines by DM on a logarithmic scale. MIN of --dm-lim must be "
                   "> 0; trials below it (incl. DM 0) get the lowest color.")
@click.option("--alpha", type=float, default=0.05, show_default=True,
              help="Line transparency.")
@click.option("--workers", type=int, default=DEFAULT_WORKERS, show_default=True,
              help="Processes decompressing HDF5 chunks in parallel. Each holds "
                   "~1-2 chunks in RAM at once (~0.9 GB uncompressed per chunk "
                   "for the full-size combined file).")
@click.option("--title", type=str, default=None,
              help="Plot title, used exactly as given (quote it; matplotlib mathtext "
                   "like '$b$' works; \"\" for no title). Default: built from the "
                   "selection and number of pointings.")
@click.option("--dpi", type=float, default=150, show_default=True,
              help="Figure resolution.")
@click.option("--output-plot", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Save the plot to this file (default: display).")
@click.option("--output-npz", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Also save median, freqs, dms, counts and the selection to this "
                   ".npz, which can be passed back in as INPUT_FILE to replot.")
def main(input_file, dec_center, dec_radius, zenith, b_center, b_radius, maxdm_cap,
         map_v1_3_path, map_v2_0_path, xlim, ylim, dm_lim, alpha, workers, dpi,
         output_plot, output_npz, title, colorbar_log):
    """
    Plot, for every DM trial, the median over pointings of the mean over
    days of the rednoise medians vs frequency, for a selection of pointings.
    Default selection: Dec 90 +/- 5. Lines are colored by trial DM.

    INPUT_FILE is either the combined medians .h5 file made by
    combine_rednoise_medians.py, or a .npz saved earlier with --output-npz,
    which just redraws that plot (no h5 reading; selection options not allowed).
    """
    try:
        _check_limits("xlim", *xlim, log=True)
        _check_limits("ylim", *ylim, log=True)
        if dm_lim is not None:
            _check_limits("dm-lim", *dm_lim, log=colorbar_log)
    except ValueError as e:
        if colorbar_log and "dm-lim" in str(e):
            e = ValueError(f"{e} (--colorbar-log needs a --dm-lim MIN above 0)")
        raise click.UsageError(str(e))

    if _is_npz(input_file):
        used = [name for name, v in (("--dec-center", dec_center), ("--dec-radius", dec_radius),
                                     ("--zenith", zenith or None), ("--b-center", b_center),
                                     ("--b-radius", b_radius), ("--output-npz", output_npz))
                if v is not None]
        if used:
            raise click.UsageError(f"{', '.join(used)} can't be used when INPUT_FILE is an "
                                   f".npz: its data was already selected when it was saved.")
        try:
            fig, _, _ = plot_from_npz(input_file, output_path=output_plot, alpha=alpha,
                                      dpi=dpi, xlim=xlim, ylim=ylim, dm_lim=dm_lim,
                                      title=title, colorbar_log=colorbar_log)
        except (ValueError, OSError) as e:
            raise click.ClickException(str(e))
        plt.close(fig)
        return

    try:
        selection = resolve_selection(zenith, dec_center, dec_radius, b_center, b_radius)
    except ValueError as e:
        raise click.UsageError(str(e))
    print(f"Selection: {selection.label}", flush=True)

    map_v1_3 = load_pointing_map(map_v1_3_path)
    map_v2_0 = load_pointing_map(map_v2_0_path)

    res = rednoise_vs_dm(input_file, map_v1_3, map_v2_0, selection=selection,
                         maxdm_cap=maxdm_cap, workers=workers)

    if output_npz:
        save_results(output_npz, res, selection)
        print(f"Arrays saved to {output_npz}", flush=True)

    fig, _, _ = plot_rednoise_vs_dm(res["median"], res["freqs"], res["dms"], alpha=alpha,
                                    n_pointings=res["n_pointings"], dpi=dpi,
                                    output_path=output_plot,
                                    selection_label=selection.label,
                                    xlim=xlim, ylim=ylim, dm_lim=dm_lim, title=title,
                                    colorbar_log=colorbar_log)
    plt.close(fig)


if __name__ == "__main__":
    main()
