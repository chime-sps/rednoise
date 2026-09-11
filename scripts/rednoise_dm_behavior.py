#!/usr/bin/env python3
"""
compute_rednoise_dm_info.py
===========================
Computes per-pointing DM statistics from combined_medians.h5 without loading
the full array into RAM.  Processes rows in chunks and accumulates statistics
incrementally.

Output (rednoise_dm_info.npz):
    info              : (N, 7) array -- ra, dec, year, month, day,
                                        mins_match (bool), threshold_dm_idx
    median_across_dms : (N, max_n_freq) float32
    std_across_dms    : (N, max_n_freq) float32
    min_across_dms    : (N, max_n_freq) float32
    max_across_dms    : (N, max_n_freq) float32

Usage
-----
    python compute_rednoise_dm_info.py combined_medians.h5
    python compute_rednoise_dm_info.py combined_medians.h5 --max-rows 1000
    python compute_rednoise_dm_info.py combined_medians.h5 --chunk-size 256
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


CHUNK_SIZE = 128   # rows loaded at once; tune based on available RAM
                   # RAM per chunk ~ CHUNK_SIZE * 16799 * 51 * 4 bytes
                   # e.g. 128 rows ~ 128 * 16799 * 51 * 4 = ~440 MB


def process_chunk(chunk_medians, chunk_n_dm, chunk_n_freq, max_n_freq):
    """
    Compute per-row statistics for a chunk of medians arrays.

    Parameters
    ----------
    chunk_medians : (chunk_size, max_n_dm, max_n_freq) float32
    chunk_n_dm    : (chunk_size,) int   -- valid DM count per row
    chunk_n_freq  : (chunk_size,) int   -- valid freq count per row
    max_n_freq    : int                 -- padded freq axis length

    Returns
    -------
    mins_match        : (chunk_size,) bool
    threshold_dm_idx  : (chunk_size,) int
    median_across_dms : (chunk_size, max_n_freq) float32
    std_across_dms    : (chunk_size, max_n_freq) float32
    min_across_dms    : (chunk_size, max_n_freq) float32
    max_across_dms    : (chunk_size, max_n_freq) float32
    """
    cs = len(chunk_medians)

    mins_match_arr       = np.zeros(cs, dtype=bool)
    threshold_dm_idx_arr = np.zeros(cs, dtype=np.int32)
    median_out           = np.zeros((cs, max_n_freq), dtype=np.float32)
    std_out              = np.zeros((cs, max_n_freq), dtype=np.float32)
    min_out              = np.zeros((cs, max_n_freq), dtype=np.float32)
    max_out              = np.zeros((cs, max_n_freq), dtype=np.float32)

    for j in range(cs):
        nd = int(chunk_n_dm[j])
        nf = int(chunk_n_freq[j])

        # Slice to valid (unpadded) region only -- excludes -1 sentinel values.
        valid = chunk_medians[j, :nd, :nf].astype(np.float64)  # (nd, nf)

        # mins_match: check whether the minimum across all freq bins equals
        # the minimum excluding the first 5 bins (index < 5 may be RFI-affected).
        # We compare per DM row (axis=-1 = freq axis).
        min_all  = np.min(valid,       axis=-1)   # (nd,)
        min_from5 = np.min(valid[:, 5:], axis=-1)  # (nd,)
        row_mins_match = (min_all == min_from5)     # (nd,)

        if np.all(row_mins_match):
            mins_match_arr[j]       = True
            threshold_dm_idx_arr[j] = 0
        else:
            mins_match_arr[j]       = False
            threshold_dm_idx_arr[j] = int(np.argmax(~row_mins_match))

        # Statistics across the DM axis (axis=0), over valid region only.
        # Results are written into the first nf slots; remaining slots stay 0.
        median_out[j, :nf] = np.median(valid, axis=0).astype(np.float32)
        std_out[j,    :nf] = np.std(valid,    axis=0).astype(np.float32)
        min_out[j,    :nf] = np.min(valid,    axis=0).astype(np.float32)
        max_out[j,    :nf] = np.max(valid,    axis=0).astype(np.float32)

    return (mins_match_arr, threshold_dm_idx_arr,
            median_out, std_out, min_out, max_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_file", help="Path to combined_medians.h5")
    parser.add_argument("--output", default="rednoise_dm_info.npz",
                        help="Output npz file (default: rednoise_dm_info.npz)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Only process the first N rows (for testing)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Rows per chunk (default: {CHUNK_SIZE}). "
                             f"Each chunk uses ~chunk_size * 16799 * 51 * 4 bytes RAM.")
    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        sys.exit(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        # FIX: use the minimum length across all datasets rather than trusting
        # medians.shape[0] alone.  If the process was killed mid-batch, medians
        # may have been resized before ra/dec/etc. were updated, leaving the
        # datasets out of sync by one batch (500 rows in the observed case).
        total_rows = min(
            h5["medians"].shape[0],
            h5["ra"].shape[0],
            h5["dec"].shape[0],
            h5["year"].shape[0],
            h5["n_dm"].shape[0],
            h5["n_freq"].shape[0],
        )
        max_n_freq = int(h5.attrs["max_n_freq"])   # padded freq axis size

        N = min(args.max_rows, total_rows) if args.max_rows else total_rows
        if args.max_rows:
            print(f"{total_rows} consistent rows, limiting to first {N}.", flush=True)
        else:
            print(f"{N} total rows.", flush=True)
        print(f"Chunk size: {args.chunk_size} rows  "
              f"(~{args.chunk_size * 16799 * 51 * 4 / 1e6:.0f} MB per chunk)",
              flush=True)

        # Load small metadata arrays fully -- these are tiny.
        ra_arr    = h5["ra"][:N]
        dec_arr   = h5["dec"][:N]
        year_arr  = h5["year"][:N]
        month_arr = h5["month"][:N]
        day_arr   = h5["day"][:N]
        n_dm_arr  = h5["n_dm"][:N]
        # FIX: read n_freq from the stored dataset, which holds the actual
        # (unpadded) number of valid frequency bins for each pointing.
        # Do NOT use medians[i].shape[1] -- that always returns max_n_freq
        # (the padded size, 51), not the number of valid bins (e.g. 31).
        n_freq_arr = h5["n_freq"][:N]

        # Pre-allocate output arrays.
        output_stats     = np.zeros((N, 7),           dtype=np.float64)
        median_across_dms = np.zeros((N, max_n_freq), dtype=np.float32)
        std_across_dms    = np.zeros((N, max_n_freq), dtype=np.float32)
        min_across_dms    = np.zeros((N, max_n_freq), dtype=np.float32)
        max_across_dms    = np.zeros((N, max_n_freq), dtype=np.float32)

        # Fill metadata columns now (no need to revisit these per chunk).
        output_stats[:, 0] = ra_arr
        output_stats[:, 1] = dec_arr
        output_stats[:, 2] = year_arr
        output_stats[:, 3] = month_arr
        output_stats[:, 4] = day_arr

        # Process medians in chunks to keep RAM bounded.
        n_chunks = int(np.ceil(N / args.chunk_size))
        for chunk_idx in range(n_chunks):
            start = chunk_idx * args.chunk_size
            end   = min(start + args.chunk_size, N)
            print(f"  Chunk {chunk_idx + 1}/{n_chunks}: rows {start}–{end-1}",
                  flush=True)

            # FIX: wrap the chunk read in a try/except.  A corrupted HDF5
            # chunk (e.g. from a mid-write kill) raises OSError: inflate()
            # failed.  When that happens fall back to row-by-row reads so we
            # can skip only the bad rows rather than aborting the whole run.
            try:
                chunk_medians = h5["medians"][start:end]
                good_rows = list(range(start, end))
            except OSError as e:
                print(f"  [WARN] Chunk read failed ({e}); "
                      f"falling back to row-by-row reads.", flush=True)
                rows = []
                good_rows = []
                for row in range(start, end):
                    try:
                        rows.append(h5["medians"][row:row+1])
                        good_rows.append(row)
                    except OSError:
                        print(f"    [SKIP] Corrupted row {row}", flush=True)
                if not rows:
                    print(f"  [WARN] All rows in chunk {chunk_idx+1} skipped.",
                          flush=True)
                    continue
                chunk_medians = np.concatenate(rows, axis=0)

            local_n_dm   = n_dm_arr[good_rows]
            local_n_freq = n_freq_arr[good_rows]

            (mins_match, threshold_dm_idx,
             median_c, std_c, min_c, max_c) = process_chunk(
                chunk_medians,
                local_n_dm,
                local_n_freq,
                max_n_freq,
            )

            for out_idx, row in enumerate(good_rows):
                output_stats[row, 5]        = float(mins_match[out_idx])
                output_stats[row, 6]        = threshold_dm_idx[out_idx]
                median_across_dms[row]      = median_c[out_idx]
                std_across_dms[row]         = std_c[out_idx]
                min_across_dms[row]         = min_c[out_idx]
                max_across_dms[row]         = max_c[out_idx]

            # Explicitly free the chunk to avoid accumulation between iterations.
            del chunk_medians

    print(f"Saving to {args.output} ...", flush=True)
    np.savez_compressed(
        args.output,
        info              = output_stats,
        median_across_dms = median_across_dms,
        std_across_dms    = std_across_dms,
        min_across_dms    = min_across_dms,
        max_across_dms    = max_across_dms,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
