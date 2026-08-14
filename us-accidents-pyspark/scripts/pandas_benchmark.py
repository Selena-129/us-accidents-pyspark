"""Three-run, full-dataset Pandas benchmark for the US Accidents project.

The workload reads the raw CSV, validates the required fields, and produces
state counts, year counts, and severity counts.
"""

import argparse
import gc
import json
import platform
import resource
import statistics
import time
from collections import Counter
from pathlib import Path

import pandas as pd


EXPECTED_ROWS = 7_728_394
REQUIRED_COLUMNS = ["State", "Severity", "Start_Time"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/Users/theng/Downloads/US_Accidents_March23.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/theng/Downloads/pandas_benchmark_results",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def peak_memory_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def add_counts(target, series):
    target.update({str(key): int(value) for key, value in series.items()})


def run_once(input_path, output_dir, run_number, chunk_size):
    state_counts = Counter()
    year_counts = Counter()
    severity_counts = Counter()
    input_rows = 0
    valid_rows = 0

    start = time.perf_counter()
    for chunk in pd.read_csv(
        input_path,
        usecols=REQUIRED_COLUMNS,
        chunksize=chunk_size,
        dtype={"State": "string", "Severity": "Int8", "Start_Time": "string"},
    ):
        input_rows += len(chunk)
        timestamps = pd.to_datetime(chunk["Start_Time"], format="mixed", errors="coerce")
        valid = (
            chunk["State"].notna()
            & chunk["Severity"].between(1, 4)
            & timestamps.notna()
        )
        part = chunk.loc[valid, ["State", "Severity"]]
        years = timestamps.loc[valid].dt.year
        valid_rows += len(part)

        add_counts(state_counts, part["State"].value_counts(sort=False))
        add_counts(year_counts, years.value_counts(sort=False))
        add_counts(severity_counts, part["Severity"].value_counts(sort=False))

        del chunk, timestamps, valid, part, years

    scan_and_aggregate_seconds = time.perf_counter() - start

    state = pd.DataFrame(
        sorted(state_counts.items(), key=lambda item: (-item[1], item[0])),
        columns=["State", "Accident_Count"],
    )
    year = pd.DataFrame(
        sorted(((int(k), v) for k, v in year_counts.items())),
        columns=["Year", "Accident_Count"],
    )
    severity = pd.DataFrame(
        sorted(((int(k), v) for k, v in severity_counts.items())),
        columns=["Severity", "Accident_Count"],
    )

    write_start = time.perf_counter()
    run_dir = output_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state.to_csv(run_dir / "state_counts.csv", index=False)
    year.to_csv(run_dir / "year_counts.csv", index=False)
    severity.to_csv(run_dir / "severity_counts.csv", index=False)
    write_seconds = time.perf_counter() - write_start
    total_seconds = scan_and_aggregate_seconds + write_seconds

    reconciled = (
        input_rows == EXPECTED_ROWS
        and valid_rows == EXPECTED_ROWS
        and int(state["Accident_Count"].sum()) == EXPECTED_ROWS
        and int(year["Accident_Count"].sum()) == EXPECTED_ROWS
        and int(severity["Accident_Count"].sum()) == EXPECTED_ROWS
    )
    result = {
        "run": run_number,
        "input_rows": input_rows,
        "valid_rows": valid_rows,
        "scan_and_aggregate_seconds": round(scan_and_aggregate_seconds, 4),
        "write_seconds": round(write_seconds, 4),
        "total_seconds": round(total_seconds, 4),
        "peak_process_memory_mb": round(peak_memory_mb(), 2),
        "reconciliation_status": "PASS" if reconciled else "FAIL",
    }
    with (run_dir / "run_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    args = arguments()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run_number in range(1, args.runs + 1):
        gc.collect()
        print(f"Starting Pandas run {run_number} of {args.runs}...")
        result = run_once(
            input_path, output_dir, run_number, args.chunk_size
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "benchmark_runs.csv", index=False)
    experiment = {
        "framework": "Pandas",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "Apple M1 (recorded from system)",
        "input": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "columns_used": REQUIRED_COLUMNS,
        "chunk_size": args.chunk_size,
        "runs": args.runs,
        "median_scan_and_aggregate_seconds": round(
            statistics.median(r["scan_and_aggregate_seconds"] for r in results), 4
        ),
        "median_total_seconds": round(
            statistics.median(r["total_seconds"] for r in results), 4
        ),
        "all_runs_reconciled": all(
            r["reconciliation_status"] == "PASS" for r in results
        ),
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment, handle, indent=2)

    print(f"Benchmark complete: {output_dir}")
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
