#!/usr/bin/env python3
"""
combine_medians.py
==================
Walks a directory tree of the form:
    {year}/{month}/{day}/{RA}_{Dec}/medians.npz

Each medians.npz contains an array of shape (1, n_dm, n_freq_bins).
Because different files may have different n_dm and n_freq_bins, all arrays
are padded to the global maximum along each axis with -1 (a sentinel flag).

Output HDF5 datasets (all length N = number of pointings):
    medians      : float32, shape (N, n_dm_max, n_freq_max)  -- -1 where padded
    scale        : float32, shape (N, n_freq_max)  -- -1 where padded
    n_dm         : int32,   shape (N,)  -- actual n_dm for each pointing
    n_freq       : int32,   shape (N,)  -- actual n_freq_bins for each pointing
    ra           : float64, shape (N,)
    dec          : float64, shape (N,)
    year         : int16,   shape (N,)
    month        : uint8,   shape (N,)
    day          : uint8,   shape (N,)

Each medians.npz also carries a 'scale' array.  In practice it is stored
redundantly -- as a 2D (n_dm, n_freq) or 3D (1, n_dm, n_freq) array whose
rows/planes are all identical -- but only one n_freq-length copy of it is
meaningful, so it is reduced (scale[0] or scale[0][0]) down to a 1D array
of length n_freq before being written out, padded to n_freq_max with -1
exactly like the medians array is. Use n_freq[i] to slice out the valid
portion of scale[i], the same way it is used to slice medians[i].

Features
--------
- Multithreaded file loading: many worker threads read npz files in parallel
  while a single writer thread flushes completed batches to HDF5.
- Variable-shape support: arrays are padded to the global max shape with -1.
- Streams to disk in batches -- RAM usage stays bounded.
- Resumable: re-running the same command skips already-written files.
- Live scan counter appears immediately on slow network filesystems.

Usage
-----
    python combine_medians.py /path/to/root -o combined_medians.h5
    python combine_medians.py /path/to/root -o combined_medians.h5  # resumes
    python combine_medians.py /path/to/root -o combined_medians.h5 \\
        --workers 32 --batch-size 500
    python combine_medians.py /path/to/root -o test_run.h5 --n-files 10  # quick test
"""

import argparse
import sys
import threading
import queue
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm is required.\n  pip install tqdm")

try:
    import h5py
except ImportError:
    sys.exit("h5py is required.\n  pip install h5py")


DEFAULT_BATCH_SIZE = 500
DEFAULT_WORKERS    = 16
WINDOW_MULTIPLIER  = 4   # keep ~4x workers jobs in flight; see iter_completed_windowed()


# ---------------------------------------------------------------------------
# Bounded-concurrency executor driver
# ---------------------------------------------------------------------------

def iter_completed_windowed(executor: ThreadPoolExecutor, fn, records, window_size: int):
    """
    Submit fn(record) for each record in `records` to `executor`, keeping at
    most `window_size` jobs in flight at once, and yield each result as soon
    as it completes (completion order, not submission order).

    This is a bounded-memory alternative to the common
        futures = {executor.submit(fn, r): r for r in records}
        for fut in as_completed(futures): ...
    pattern, which submits every record to the executor up front. That's
    fine at small-to-moderate scale, but each in-flight Future carries real
    bookkeeping (a lock, a condition variable, callback list, ...) -- on the
    order of 1-2 KB apiece -- so for a run with hundreds of thousands to
    millions of records, submitting everything at once can cost hundreds of
    MB to multiple GB just in that bookkeeping, on top of the records
    themselves and regardless of how much actual file/array data is in
    flight. Since only `executor._max_workers` jobs ever run concurrently
    anyway, a window of a few times that is enough to keep every worker
    thread fed, with the rest of `records` left untouched (just an
    iterator position, not materialised as Futures) until there's room.
    """
    pending_iter = iter(records)
    in_flight: dict = {}

    def submit_one() -> bool:
        try:
            record = next(pending_iter)
        except StopIteration:
            return False
        in_flight[executor.submit(fn, record)] = record
        return True

    for _ in range(window_size):
        if not submit_one():
            break

    while in_flight:
        done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            del in_flight[fut]
            yield fut.result()
            submit_one()


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

class _ScanLimitReached(Exception):
    """Internal control-flow signal: stop the directory walk early."""


def find_npz_paths(root: Path, limit: int | None = None) -> list:
    """
    Return a list of (path_str, year, month, day, ra, dec) for every
    medians.npz found under root, descending only into 4-digit year dirs.
    Shows a live counter during the scan.

    If `limit` is given, the walk stops as soon as `limit` files have been
    found -- it does not keep scanning the rest of the tree first and
    truncate afterwards, so on a large/slow filesystem a small `limit`
    keeps the scan itself fast, not just the loading step that follows it.
    """
    year_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 4
    )
    if not year_dirs:
        print(f"  [WARN] No year directories found under {root}", file=sys.stderr)
        return []

    found = []
    try:
        with tqdm(
            desc="Scanning",
            unit="file",
            dynamic_ncols=True,
            bar_format="{desc}: {n} {unit} found [{elapsed}]",
        ) as scan_bar:
            for year_dir in year_dirs:
                for month_dir in sorted(year_dir.iterdir()):
                    if not month_dir.is_dir():
                        continue
                    scan_bar.set_postfix_str(
                        f"{year_dir.name}/{month_dir.name}", refresh=True
                    )
                    for day_dir in sorted(month_dir.iterdir()):
                        if not day_dir.is_dir():
                            continue
                        for ra_dec_dir in sorted(day_dir.iterdir()):
                            if not ra_dec_dir.is_dir():
                                continue
                            npz_path = ra_dec_dir / "medians.npz"
                            if not npz_path.exists():
                                continue
                            try:
                                year  = int(year_dir.name)
                                month = int(month_dir.name)
                                day   = int(day_dir.name)
                                ra_str, dec_str = ra_dec_dir.name.rsplit("_", 1)
                                ra  = float(ra_str)
                                dec = float(dec_str)
                            except (ValueError, IndexError) as exc:
                                tqdm.write(f"  [SKIP] Bad path {npz_path}: {exc}")
                                continue
                            found.append((str(npz_path), year, month, day, ra, dec))
                            scan_bar.update(1)
                            if limit is not None and len(found) >= limit:
                                raise _ScanLimitReached
    except _ScanLimitReached:
        pass
    return found


# ---------------------------------------------------------------------------
# Per-file loader (runs in worker threads)
# ---------------------------------------------------------------------------

def reduce_scale(arr: np.ndarray) -> np.ndarray:
    """
    Reduce a 'scale' array down to the 1D (n_freq,) array it actually carries.

    'scale' may be stored as 1D (n_freq,), 2D (n_dm, n_freq), or 3D
    (1, n_dm, n_freq). In the 2D/3D cases every row/plane along the leading
    axes is identical, so repeatedly indexing the first element -- scale[0]
    for 2D, scale[0][0] for 3D -- recovers the single meaningful copy without
    losing any information.
    """
    while arr.ndim > 1:
        arr = arr[0]
    return arr


def load_one(record: tuple) -> tuple | None:
    """
    Load a single medians.npz.  Returns
        (path_str, year, month, day, ra, dec, array_2d, scale_1d)
    where array_2d has shape (n_dm, n_freq) after squeezing the leading 1-dim,
    scale_1d has shape (n_freq,) after reduce_scale(), or None on failure
    (failure message already printed via tqdm.write).

    This function is safe to call from multiple threads simultaneously because
    it only does file I/O and numpy operations -- no shared mutable state.
    """
    path_str, year, month, day, ra, dec = record
    try:
        with np.load(path_str) as f:
            if "medians" in f:
                arr = f["medians"]
            else:
                key = list(f.keys())[0]
                arr = f[key]
                tqdm.write(f"  [WARN] {path_str}: key '{key}' used instead of 'medians'")

            # Shape is (1, n_dm, n_freq_bins) -- squeeze the leading dimension.
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]          # now (n_dm, n_freq_bins)
            elif arr.ndim == 2:
                pass                  # already (n_dm, n_freq_bins)
            else:
                tqdm.write(
                    f"  [SKIP] {path_str}: unexpected shape {arr.shape}, "
                    f"expected (1, n_dm, n_freq) or (n_dm, n_freq)"
                )
                return None

            arr = arr.astype(np.float32)

            n_freq = arr.shape[1]
            # The rednoise files store this under the key "scales" (plural).
            # Fall back to "scale" too, in case an older/other file uses the
            # singular form.
            scale_key = "scales" if "scales" in f else ("scale" if "scale" in f else None)
            if scale_key is not None:
                scale = reduce_scale(np.asarray(f[scale_key])).astype(np.float32)
                if scale.ndim != 1:
                    tqdm.write(
                        f"  [WARN] {path_str}: could not reduce '{scale_key}' to 1D "
                        f"(got shape {scale.shape}); storing as all -1"
                    )
                    scale = np.full(n_freq, -1.0, dtype=np.float32)
                elif scale.shape[0] != n_freq:
                    tqdm.write(
                        f"  [WARN] {path_str}: '{scale_key}' length {scale.shape[0]} != "
                        f"n_freq {n_freq}; truncating/padding to match"
                    )
                    fixed = np.full(n_freq, -1.0, dtype=np.float32)
                    m = min(n_freq, scale.shape[0])
                    fixed[:m] = scale[:m]
                    scale = fixed
            else:
                tqdm.write(f"  [WARN] {path_str}: no 'scales' key found; storing as all -1")
                scale = np.full(n_freq, -1.0, dtype=np.float32)
    except Exception as exc:
        tqdm.write(f"  [SKIP] Cannot load {path_str}: {exc}")
        return None

    return (path_str, year, month, day, ra, dec, arr, scale)


# ---------------------------------------------------------------------------
# Shape discovery: scan all files to find global max n_dm and n_freq
# ---------------------------------------------------------------------------

def discover_max_shape(pending: list, workers: int) -> tuple[int, int]:
    """
    Read every pending file's array shape to find the global maximum n_dm
    and n_freq_bins across all files.

    Note this is not free: numpy's .npz reader has no way to report an
    array's shape without materialising the whole array (there's no
    "read the header only" path for a zipped/compressed .npz the way
    mmap_mode='r' gives you for a plain .npy file), so each peek()
    temporarily holds one full 'medians' array in memory. Threads still
    help because the actual file read/decompression is I/O-bound and
    releases the GIL; peak memory here is roughly `workers` arrays at once,
    via iter_completed_windowed() below.
    """
    max_dm   = 0
    max_freq = 0
    lock     = threading.Lock()
    skipped  = 0

    def peek(record):
        nonlocal skipped
        path_str = record[0]
        try:
            with np.load(path_str) as f:
                key = "medians" if "medians" in f else list(f.keys())[0]
                shape = f[key].shape
                # shape is (1, n_dm, n_freq) or (n_dm, n_freq)
                if len(shape) == 3:
                    _, n_dm, n_freq = shape
                elif len(shape) == 2:
                    n_dm, n_freq = shape
                else:
                    return
            with lock:
                nonlocal max_dm, max_freq
                if n_dm   > max_dm:   max_dm   = n_dm
                if n_freq > max_freq: max_freq = n_freq
        except Exception:
            with lock:
                skipped += 1

    print("Discovering array shapes across all files ...", flush=True)
    window_size = max(workers * WINDOW_MULTIPLIER, 1)
    with tqdm(total=len(pending), desc="Shape scan", unit="file",
              dynamic_ncols=True, smoothing=0.05) as bar:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in iter_completed_windowed(ex, peek, pending, window_size):
                # peek()'s return value is unused (it reports via the shared
                # max_dm/max_freq + lock above); iterating still re-raises
                # any unexpected exception from a worker.
                bar.update(1)

    if skipped:
        tqdm.write(f"  [WARN] {skipped} file(s) could not be peeked during shape scan.")
    print(f"Global max shape: n_dm={max_dm}, n_freq={max_freq}", flush=True)
    return max_dm, max_freq


# ---------------------------------------------------------------------------
# HDF5 initialisation
# ---------------------------------------------------------------------------

def init_h5(h5: h5py.File, max_dm: int, max_freq: int,
            total: int, batch_size: int) -> None:
    """Create all resizable datasets in a brand-new HDF5 file."""
    opts    = dict(compression="gzip", compression_opts=4)
    chunk_n = min(batch_size, 256)

    # maxshape=None on every axis (not just axis 0) so n_dm/n_freq can grow
    # later too -- e.g. when resuming finds a file bigger than any seen so
    # far. fillvalue=-1 makes HDF5 auto-backfill every newly-created cell
    # (in brand-new rows AND in the new columns of already-written rows)
    # with -1, matching the padding convention used everywhere else here.
    h5.create_dataset(
        "medians",
        shape=(0, max_dm, max_freq), maxshape=(None, None, None),
        dtype=np.float32, chunks=(chunk_n, max_dm, max_freq),
        fillvalue=-1.0, **opts,
    )
    h5.create_dataset(
        "scale",
        shape=(0, max_freq), maxshape=(None, None),
        dtype=np.float32, chunks=(chunk_n, max_freq),
        fillvalue=-1.0, **opts,
    )
    for name, dtype in [
        ("n_dm",  np.int32),
        ("n_freq", np.int32),
        ("ra",    np.float64),
        ("dec",   np.float64),
        ("year",  np.int16),
        ("month", np.uint8),
        ("day",   np.uint8),
    ]:
        h5.create_dataset(
            name, shape=(0,), maxshape=(None,), dtype=dtype,
            chunks=(chunk_n,), **opts,
        )

    # Store the pad shape as metadata so readers know the sentinel convention.
    h5.attrs["max_n_dm"]         = max_dm
    h5.attrs["max_n_freq"]       = max_freq
    h5.attrs["pad_sentinel"]     = -1.0
    h5.attrs["total_expected"]   = total


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def pad_to(arr: np.ndarray, *target_shape: int) -> np.ndarray:
    """
    Pad an array up to target_shape with -1, aligned at the origin along
    every axis. Works for both the 2D (n_dm, n_freq) medians arrays and the
    1D (n_freq,) scale arrays. Returns the array unchanged if it already has
    the target shape.
    """
    if arr.shape == target_shape:
        return arr
    padded = np.full(target_shape, -1.0, dtype=np.float32)
    padded[tuple(slice(0, n) for n in arr.shape)] = arr
    return padded


def ensure_capacity(h5: h5py.File, max_dm: int, max_freq: int, batch: list) -> tuple[int, int]:
    """
    Grow the 'medians'/'scale' datasets if any record in this batch needs a
    bigger n_dm or n_freq than currently allocated, and return the
    (possibly updated) (max_dm, max_freq).

    This matters most on resume: the max shape may have been fixed by an
    earlier, smaller run (e.g. a --n-files test) that never saw the largest
    file. Both datasets were created with fillvalue=-1 and fully unlimited
    maxshape, so resizing them here automatically backfills every
    newly-created cell -- in brand-new rows and in the newly-added columns
    of rows already on disk -- with -1, exactly like pad_to() does for a
    single array. Without this, a later batch containing a bigger array
    would fail to fit into the previously fixed padded shape.
    """
    if not batch:
        return max_dm, max_freq

    arrays = [rec[6] for rec in batch]
    scales = [rec[7] for rec in batch]

    needed_dm   = max([max_dm]   + [a.shape[0] for a in arrays])
    needed_freq = max([max_freq] + [a.shape[1] for a in arrays] + [s.shape[0] for s in scales])

    if needed_dm > max_dm or needed_freq > max_freq:
        tqdm.write(
            f"  [INFO] Growing padded shape: n_dm {max_dm}->{needed_dm}, "
            f"n_freq {max_freq}->{needed_freq} (found a larger file than "
            f"any seen so far)"
        )
        n_rows = h5["medians"].shape[0]
        h5["medians"].resize((n_rows, needed_dm, needed_freq))
        h5["scale"].resize((n_rows, needed_freq))
        h5.attrs["max_n_dm"]   = needed_dm
        h5.attrs["max_n_freq"] = needed_freq
        max_dm, max_freq = needed_dm, needed_freq

    return max_dm, max_freq


# ---------------------------------------------------------------------------
# Batch writer
# ---------------------------------------------------------------------------

def flush_batch(h5: h5py.File, batch: list, max_dm: int, max_freq: int) -> None:
    """Pad and append one batch of loaded records to the open HDF5 file."""
    if not batch:
        return

    paths, years, months, days, ras, decs, arrays, scales = zip(*batch)
    n = len(batch)

    # Pad each array to the global max shape.
    padded = np.stack(
        [pad_to(a, max_dm, max_freq) for a in arrays]
    ).astype(np.float32)                                  # (n, max_dm, max_freq)

    padded_scale = np.stack(
        [pad_to(s, max_freq) for s in scales]
    ).astype(np.float32)                                  # (n, max_freq)

    actual_dms   = np.array([a.shape[0] for a in arrays], dtype=np.int32)
    actual_freqs = np.array([a.shape[1] for a in arrays], dtype=np.int32)

    for name, arr in [
        ("medians", padded),
        ("scale",   padded_scale),
        ("n_dm",    actual_dms),
        ("n_freq",  actual_freqs),
        ("ra",      np.array(ras,    dtype=np.float64)),
        ("dec",     np.array(decs,   dtype=np.float64)),
        ("year",    np.array(years,  dtype=np.int16)),
        ("month",   np.array(months, dtype=np.uint8)),
        ("day",     np.array(days,   dtype=np.uint8)),
    ]:
        ds      = h5[name]
        old_len = ds.shape[0]
        ds.resize(old_len + n, axis=0)
        ds[old_len:] = arr

    # Checkpoint: record which paths are now safely on disk.
    str_dtype = h5py.special_dtype(vlen=str)
    if "_done" not in h5:
        h5.create_dataset("_done", shape=(0,), maxshape=(None,), dtype=str_dtype)
    done_ds = h5["_done"]
    old_len = done_ds.shape[0]
    done_ds.resize(old_len + n, axis=0)
    done_ds[old_len:] = list(paths)

    h5.flush()


def load_done_set(h5: h5py.File) -> set:
    if "_done" not in h5:
        return set()
    # BUG FIX: h5py returns variable-length strings as bytes in Python 3 with
    # older h5py/HDF5 builds.  Decode unconditionally so the set always holds
    # plain str objects that compare equal to the path strings in all_paths.
    # Without this fix every entry looks "not done" on every resume because
    # b"/path/..." != "/path/..." -- so the resume never skips anything and
    # always restarts from the same point regardless of prior progress.
    raw = h5["_done"][:].tolist()
    return {v.decode() if isinstance(v, bytes) else v for v in raw}


# ---------------------------------------------------------------------------
# Main combine routine
# ---------------------------------------------------------------------------

def combine(root: Path, output: Path, batch_size: int, workers: int,
            cli_max_dm: int | None = None, cli_max_freq: int | None = None,
            n_files: int | None = None) -> None:
    root = root.resolve()
    print(f"Scanning {root} ...", flush=True)

    if n_files is not None:
        print(f"--n-files {n_files} given -- scan will stop early once found.", flush=True)
    all_paths = find_npz_paths(root, limit=n_files)
    if not all_paths:
        sys.exit("No medians.npz files found - check the root path.")

    total = len(all_paths)
    print(f"Found {total} pointing(s).", flush=True)

    file_mode = "a" if output.exists() else "w"
    if file_mode == "a":
        print("Output file exists -- resuming from previous run.", flush=True)

    with h5py.File(output, file_mode) as h5:

        # ── Resume: filter out already-written paths ────────────────────────
        done    = load_done_set(h5)
        pending = [r for r in all_paths if r[0] not in done]
        if done:
            print(f"  {len(done)} already written, {len(pending)} remaining.",
                  flush=True)
        if not pending:
            print("Nothing to do -- all files already written.")
            return

        # ── Determine max shape ─────────────────────────────────────────────
        # Priority: (1) shape already in HDF5 file, (2) CLI --max-dm/--max-freq,
        # (3) full shape scan across all pending files.
        if "medians" in h5:
            # Datasets already initialised -- read shape from file attributes.
            max_dm   = int(h5.attrs["max_n_dm"])
            max_freq = int(h5.attrs["max_n_freq"])
            print(f"Resuming with stored max shape: n_dm={max_dm}, n_freq={max_freq}",
                  flush=True)
            if "scale" not in h5:
                # Output file was created before 'scale' was tracked -- add it
                # now so resumed runs still get it for all remaining pointings.
                n_existing = h5["medians"].shape[0]
                chunk_n = min(batch_size, 256)
                scale_ds = h5.create_dataset(
                    "scale",
                    shape=(n_existing, max_freq), maxshape=(None, None),
                    dtype=np.float32, chunks=(chunk_n, max_freq),
                    compression="gzip", compression_opts=4, fillvalue=-1.0,
                )
                scale_ds[:] = -1.0  # backfill: unknown for already-written rows
                print(
                    "  [INFO] Added 'scale' dataset to existing output file "
                    f"(backfilled {n_existing} prior row(s) with -1).",
                    flush=True,
                )
        elif cli_max_dm is not None and cli_max_freq is not None:
            # Shape supplied on the command line -- use it directly and
            # initialise the datasets without scanning any files.
            max_dm, max_freq = cli_max_dm, cli_max_freq
            print(
                f"Using CLI-supplied max shape: n_dm={max_dm}, n_freq={max_freq}",
                flush=True,
            )
            init_h5(h5, max_dm, max_freq, total, batch_size)
        else:
            # No prior knowledge -- scan all files to find the global maximum.
            max_dm, max_freq = discover_max_shape(pending, workers)
            init_h5(h5, max_dm, max_freq, total, batch_size)

        # ── Multithreaded load + single-threaded HDF5 write ─────────────────
        #
        # Architecture:
        #   ThreadPoolExecutor loads files via load_one(). The main thread
        #   consumes completed results (in completion order), accumulates
        #   them into batches, and flushes each batch to HDF5.
        #
        # Why not write from threads?  h5py's default build is not thread-safe
        # for concurrent writes to the same file.  Keeping all HDF5 operations
        # on the main thread avoids corruption without needing HDF5's optional
        # thread-safety build.
        #
        # Why ThreadPoolExecutor and not multiprocessing?  np.load is I/O-bound
        # (network filesystem reads), so Python's GIL is released during the
        # actual read.  Threads give us parallelism without the overhead of
        # spawning processes or pickling data across process boundaries.
        #
        # Why iter_completed_windowed() instead of submitting `pending`
        # (every remaining file) to the executor up front?  Only `workers`
        # jobs ever run concurrently anyway, but the naive
        # {executor.submit(fn, r): r for r in pending} + as_completed(...)
        # pattern still materialises one Future per pending file immediately.
        # Each Future carries non-trivial bookkeeping, so on a run with
        # hundreds of thousands to millions of pending files that upfront
        # submission alone can cost hundreds of MB to multiple GB, on top of
        # everything else -- for no benefit, since the extra Futures just
        # sit unstarted. Keeping a small window in flight avoids that
        # entirely, independent of how many files are pending overall.

        skipped = 0
        batch   = []

        bar = tqdm(
            total=total,
            initial=len(done),
            desc="Loading",
            unit="file",
            dynamic_ncols=True,
            smoothing=0.05,
        )

        window_size = max(workers * WINDOW_MULTIPLIER, 1)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in iter_completed_windowed(executor, load_one, pending, window_size):
                bar.update(1)

                if result is None:   # load_one already reported the error
                    skipped += 1
                    continue

                path_str, year, month, day, ra, dec, arr, scale = result
                bar.set_postfix_str(f"RA={ra:+.2f} Dec={dec:+.2f}", refresh=False)
                batch.append((path_str, year, month, day, ra, dec, arr, scale))

                if len(batch) >= batch_size:
                    max_dm, max_freq = ensure_capacity(h5, max_dm, max_freq, batch)
                    flush_batch(h5, batch, max_dm, max_freq)
                    batch.clear()

        # Flush any remaining records after all futures are done.
        max_dm, max_freq = ensure_capacity(h5, max_dm, max_freq, batch)
        flush_batch(h5, batch, max_dm, max_freq)
        bar.close()

        n_written = h5["medians"].shape[0]

    print(f"\nDone. {n_written} pointings written to {output}")
    print(f"  Padded shape : (N={n_written}, n_dm={max_dm}, n_freq={max_freq})")
    print(f"  File size    : {output.stat().st_size / 1e9:.3f} GB")
    if skipped:
        print(f"  Skipped      : {skipped} file(s) due to load errors")
    print(
        "\nNOTE: use n_dm[i] and n_freq[i] to slice valid data for pointing i:\n"
        "  with h5py.File(output) as f:\n"
        "      m = f['medians'][i, :f['n_dm'][i], :f['n_freq'][i]]\n"
        "      s = f['scale'][i, :f['n_freq'][i]]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root", nargs="?", default=".",
        help="Root directory to search (default: current dir)",
    )
    parser.add_argument(
        "-o", "--output", default="combined_medians.h5",
        help="Output HDF5 file (default: combined_medians.h5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Records to buffer before each HDF5 write (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of file-loading threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--max-dm", type=int, default=None,
        help=(
            "Skip the shape-scan and use this value as max n_dm. "
            "Must be paired with --max-freq. "
            "Use when resuming a run that was killed before the first batch flush, "
            "e.g. --max-dm 16799 --max-freq 51"
        ),
    )
    parser.add_argument(
        "--max-freq", type=int, default=None,
        help="Skip the shape-scan and use this value as max n_freq_bins. Must be paired with --max-dm.",
    )
    parser.add_argument(
        "--n-files", "--n_files", dest="n_files", type=int, default=None,
        help=(
            "Stop after discovering this many medians.npz files -- useful for "
            "a quick test run on a small subset before doing a full combine."
        ),
    )
    args = parser.parse_args()

    if (args.max_dm is None) != (args.max_freq is None):
        parser.error("--max-dm and --max-freq must be supplied together or not at all.")

    if args.n_files is not None and args.n_files <= 0:
        parser.error("--n-files must be a positive integer.")

    combine(
        Path(args.root), Path(args.output),
        args.batch_size, args.workers,
        cli_max_dm=args.max_dm, cli_max_freq=args.max_freq,
        n_files=args.n_files,
    )


if __name__ == "__main__":
    main()
