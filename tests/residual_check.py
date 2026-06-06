#!/usr/bin/env python3
"""
Compare residual histories between two OpenFOAM log files.

Typical use: verify that metalPCG converges to the same residual tolerance
as vanilla PCG in the same number of iterations.

Usage:
    python3 residual_check.py log.vanilla log.metalPCG
    python3 residual_check.py log.vanilla log.metalPCG --field p --plot
"""

import re
import sys
import argparse
from pathlib import Path


def parse_residuals(log_path, field="p"):
    """
    Extract (initial_residual, final_residual, n_iterations) per solve call
    for the given field from an OpenFOAM log file.
    """
    pattern = re.compile(
        rf"Solving for {re.escape(field)},\s+"
        rf"Initial residual = ([0-9eE+\-.]+),\s+"
        rf"Final residual = ([0-9eE+\-.]+),\s+"
        rf"No Iterations (\d+)"
    )
    records = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append((
                    float(m.group(1)),
                    float(m.group(2)),
                    int(m.group(3)),
                ))
    return records


def compare(ref_log, test_log, field="p", tol=0.05):
    """
    Compare two residual histories.  Passes if:
      - Same number of solve calls
      - Each final residual within tol (relative) of reference
      - Mean iteration count within tol of reference
    """
    ref  = parse_residuals(ref_log,  field)
    test = parse_residuals(test_log, field)

    print(f"Field: {field}")
    print(f"  {ref_log}:  {len(ref)} solves")
    print(f"  {test_log}: {len(test)} solves")

    if not ref:
        sys.exit(f"ERROR: no '{field}' residuals found in {ref_log}")
    if not test:
        sys.exit(f"ERROR: no '{field}' residuals found in {test_log}")

    n = min(len(ref), len(test))
    mismatches = []
    for i, (r, t) in enumerate(zip(ref[:n], test[:n])):
        ref_final, test_final = r[1], t[1]
        if ref_final > 0:
            rel_err = abs(test_final - ref_final) / ref_final
            if rel_err > tol:
                mismatches.append((i, ref_final, test_final, rel_err))

    ref_iters  = sum(r[2] for r in ref)  / len(ref)
    test_iters = sum(t[2] for t in test) / len(test)
    iter_ratio = test_iters / ref_iters if ref_iters > 0 else float("inf")

    print(f"\n  Mean iterations/solve — ref: {ref_iters:.1f}  test: {test_iters:.1f}  "
          f"ratio: {iter_ratio:.2f}")

    if mismatches:
        print(f"\n  FAIL: {len(mismatches)} solves exceeded {tol*100:.0f}% residual tolerance:")
        for i, rf, tf, rel in mismatches[:10]:
            print(f"    solve {i:4d}: ref={rf:.2e}  test={tf:.2e}  rel_err={rel:.1%}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches)-10} more")
        return False

    if iter_ratio > 1 + tol:
        print(f"\n  WARN: test uses {iter_ratio:.2f}× more iterations than reference")

    print(f"\n  PASS — all final residuals within {tol*100:.0f}% of reference")
    return True


def plot(ref_log, test_log, field="p"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib required for --plot: pip install matplotlib")

    ref  = parse_residuals(ref_log,  field)
    test = parse_residuals(test_log, field)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.semilogy([r[0] for r in ref],  label=Path(ref_log).name,  color="tab:blue")
    ax1.semilogy([t[0] for t in test], label=Path(test_log).name, color="tab:orange",
                 linestyle="--")
    ax1.set_ylabel("Initial residual")
    ax1.legend()
    ax1.set_title(f"Field: {field}")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot([r[2] for r in ref],  label=Path(ref_log).name,  color="tab:blue")
    ax2.plot([t[2] for t in test], label=Path(test_log).name, color="tab:orange",
             linestyle="--")
    ax2.set_ylabel("Iterations")
    ax2.set_xlabel("Solve call")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref_log",  help="Reference log file (e.g. vanilla PCG)")
    parser.add_argument("test_log", help="Test log file (e.g. metalPCG)")
    parser.add_argument("--field",  default="p", help="Field name to compare (default: p)")
    parser.add_argument("--tol",    default=0.05, type=float,
                        help="Relative residual tolerance for pass/fail (default: 0.05)")
    parser.add_argument("--plot",   action="store_true", help="Plot residual histories")
    args = parser.parse_args()

    passed = compare(args.ref_log, args.test_log, field=args.field, tol=args.tol)

    if args.plot:
        plot(args.ref_log, args.test_log, field=args.field)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
