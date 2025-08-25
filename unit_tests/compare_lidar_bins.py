#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Dict, Tuple, List
import numpy as np


LOG_FILE = "verify_lidar_log.txt"

def setup_logger(log_path: str):
    """Redirect print statements to both console and log file."""
    log_fh = open(log_path, "w", buffering=1)

    class Logger:
        def write(self, msg):
            sys.__stdout__.write(msg)
            log_fh.write(msg)
        def flush(self):
            sys.__stdout__.flush()
            log_fh.flush()

    sys.stdout = Logger()
    sys.stderr = Logger()


DEFAULT_OUTPUT_ROOT   = os.path.join(os.getcwd(), "output")
DEFAULT_SOLUTION_ROOT = os.path.join(os.getcwd(), "solution", "red3")
DEFAULT_VERIFY_RSLIDAR = False  # True: rslidar128, False: lslidar128

POINT_DTYPE = np.dtype([
    ('x',         np.float32),
    ('y',         np.float32),
    ('z',         np.float32),
    ('intensity', np.float32),
    ('ring',      np.uint16),
    ('timestamp', np.uint64),   # microseconds
])

def load_bin(path: str, dtype: np.dtype = POINT_DTYPE) -> np.ndarray:
    """Load a binary .bin file as a structured numpy array with the given dtype."""
    try:
        data = np.fromfile(path, dtype=dtype)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def list_bin_files(folder: str) -> Dict[str, str]:
    """Return {timestamp_str: full_path} for all .bin files in folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    out: Dict[str, str] = {}
    for fn in os.listdir(folder):
        if fn.endswith(".bin"):
            ts = fn[:-4]  # strip .bin
            out[ts] = os.path.join(folder, fn)
    return out

def _field_stats_float(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compute simple diff stats for float arrays of same shape."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    adiff = np.abs(diff)
    return {
        "max_abs_diff": float(adiff.max(initial=0.0)),
        "mean_abs_diff": float(adiff.mean() if adiff.size else 0.0),
    }

def compare_bin_arrays(
    A: np.ndarray,
    B: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    sort_by_timestamp_then_ring: bool = True,
    max_report_indices: int = 10,
) -> Tuple[bool, str]:
    """
    Compare two structured arrays field-by-field.
    Returns (identical, human_readable_report).
    """
    lines: List[str] = []
    ok = True

    if A.dtype != B.dtype:
        ok = False
        lines.append(f"- DTYPE mismatch:\n  A: {A.dtype}\n  B: {B.dtype}")

    if A.size != B.size:
        ok = False
        lines.append(f"- POINT COUNT mismatch: A={A.size}, B={B.size}")

    # Short-circuit if either count is zero but not equal
    if A.size == 0 and B.size == 0:
        lines.append("- Both files contain 0 points.")
        return (ok, "\n".join(lines))

    # Align order if requested and possible
    if sort_by_timestamp_then_ring and ('timestamp' in A.dtype.names) and ('ring' in A.dtype.names):
        A = np.sort(A, order=('timestamp', 'ring'))
        B = np.sort(B, order=('timestamp', 'ring'))

    # Field-by-field checks (only over fields present in both)
    common_fields = [f for f in A.dtype.names if f in B.dtype.names] if A.dtype.names and B.dtype.names else []
    if not common_fields:
        lines.append("- No common fields to compare.")
        return (False, "\n".join(lines))

    # Limit comparison to min length to allow detailed stats when counts differ
    n = min(A.size, B.size)

    for f in common_fields:
        a = A[f][:n]
        b = B[f][:n]

        # NaN checks for float fields
        if np.issubdtype(a.dtype, np.floating):
            a_nan = np.isnan(a)
            b_nan = np.isnan(b)
            if a_nan.any() or b_nan.any():
                lines.append(f"- NaN counts in '{f}': A={int(a_nan.sum())}, B={int(b_nan.sum())}")
                # Treat NaN vs NaN as equal; compare finite entries only
                finite_mask = ~(a_nan | b_nan)
            else:
                finite_mask = np.ones_like(a, dtype=bool)

            eq = np.allclose(a[finite_mask], b[finite_mask], rtol=rtol, atol=atol) and np.array_equal(a_nan, b_nan)
            if not eq:
                ok = False
                stats = _field_stats_float(a[finite_mask], b[finite_mask]) if finite_mask.any() else {"max_abs_diff": float('nan'), "mean_abs_diff": float('nan')}
                # Show first few differing indices
                diff_idx = np.where(~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True))[0]
                sample_idx = ", ".join(map(str, diff_idx[:max_report_indices]))
                lines.append(
                    f"- Field '{f}' mismatch: "
                    f"max_abs_diff={stats['max_abs_diff']:.3e}, mean_abs_diff={stats['mean_abs_diff']:.3e}, "
                    f"num_diff={diff_idx.size} (showing first {min(max_report_indices, diff_idx.size)} idx: {sample_idx})"
                )
        elif np.issubdtype(a.dtype, np.integer):
            eq = np.array_equal(a, b)
            if not eq:
                ok = False
                diff_idx = np.where(a != b)[0]
                sample_idx = ", ".join(map(str, diff_idx[:max_report_indices]))
                # Provide a tiny peek at mismatched values
                pairs = ", ".join(f"{i}:A={int(a[i])},B={int(b[i])}" for i in diff_idx[:max_report_indices])
                lines.append(
                    f"- Field '{f}' mismatch: num_diff={diff_idx.size} "
                    f"(first indices: {sample_idx}); values: {pairs}"
                )
        else:
            # Fallback for unexpected dtypes: exact compare
            eq = np.array_equal(a, b)
            if not eq:
                ok = False
                lines.append(f"- Field '{f}' mismatch (dtype {a.dtype}), values differ.")

    if ok:
        lines.append("✓ Files are identical within tolerances.")

    return ok, "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Verify LiDAR .bin files (converted vs solution) by timestamp.")
    parser.add_argument("--output_root",   default=DEFAULT_OUTPUT_ROOT,   help="Path to the 'output' root folder.")
    parser.add_argument("--solution_root", default=DEFAULT_SOLUTION_ROOT, help="Path to the 'solution/red3' root folder.")
    parser.add_argument("--verify_rslidar", action="store_true", default=DEFAULT_VERIFY_RSLIDAR, help="If set, use rslidar128 subfolder; else lslidar128.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for float comparisons.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for float comparisons.")
    parser.add_argument("--no_sort", action="store_true", help="Do not sort by (timestamp, ring) before comparing.")
    parser.add_argument("--max_report_indices", type=int, default=10, help="Max differing indices to show per field.")
    args = parser.parse_args() # Setup logging before doing anything
    setup_logger(LOG_FILE)
    print(f"Writing log to {LOG_FILE}\n")



    lidar_sub = "rslidar128" if args.verify_rslidar else "lslidar128"
    output_lidar_folder   = os.path.join(args.output_root,   lidar_sub)
    solution_lidar_folder = os.path.join(args.solution_root, lidar_sub)

    print(f"Comparing folders:\n  OUTPUT  : {output_lidar_folder}\n  SOLUTION: {solution_lidar_folder}\n")

    # Build maps {timestamp_str: path}
    output_bins   = list_bin_files(output_lidar_folder)
    solution_bins = list_bin_files(solution_lidar_folder)

    print(f"Number of output lidar bins  : {len(output_bins)}")
    print(f"Number of solution lidar bins: {len(solution_bins)}")

    # Timestamp set comparison
    output_ts   = set(output_bins.keys())
    solution_ts = set(solution_bins.keys())

    missing_in_output  = sorted(solution_ts - output_ts)
    missing_in_solution = sorted(output_ts - solution_ts)

    if missing_in_output:
        print(f"\nTimestamps present in SOLUTION but missing in OUTPUT ({len(missing_in_output)}):")
        print(", ".join(missing_in_output[:20]) + (" ..." if len(missing_in_output) > 20 else ""))

    if missing_in_solution:
        print(f"\nTimestamps present in OUTPUT but missing in SOLUTION ({len(missing_in_solution)}):")
        print(", ".join(missing_in_solution[:20]) + (" ..." if len(missing_in_solution) > 20 else ""))

    common_ts = sorted(output_ts & solution_ts)
    print(f"\nCommon timestamps to compare: {len(common_ts)}")

    identical_cnt = 0
    diffs: List[str] = []
    for ts in common_ts:
        a_path = output_bins[ts]
        b_path = solution_bins[ts]
        try:
            A = load_bin(a_path)
            B = load_bin(b_path)
        except Exception as e:
            diffs.append(f"[{ts}] ERROR reading files: {e}")
            continue

        ok, report = compare_bin_arrays(
            A, B,
            rtol=args.rtol,
            atol=args.atol,
            sort_by_timestamp_then_ring=not args.no_sort,
            max_report_indices=args.max_report_indices,
        )
        if ok:
            identical_cnt += 1
        else:
            header = f"\n--- DIFF for timestamp {ts} ---\nOUTPUT:   {a_path}\nSOLUTION: {b_path}\n"
            diffs.append(header + report)

    print(f"\n=== SUMMARY ===")
    print(f"Identical files (within tolerances): {identical_cnt} / {len(common_ts)}")
    if diffs:
        print("\nDifferences found:")
        for d in diffs:
            print(d)
    else:
        print("No differences among common timestamps.")

if __name__ == "__main__":
    # If running with no args, emulate your original defaults
    if len(sys.argv) == 1:
        # Mirror your original behaviour and print CWD for convenience
        cwd = os.getcwd()
        print("The current working directory is:", cwd)
    main()
