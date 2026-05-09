# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import subprocess
import pandas as pd
import argparse
import numpy as np


BENCH_PREFIX = "cvcuda_bench_"
BENCH_OUTPUT = "out.csv"
BENCH_COMMAND = "{} {} --csv {}"
BENCH_COLNAME = "Benchmark"
BENCH_RESULTS = "bench_output.csv"
GPU_TIME_COLNAME = "GPU Time (sec)"
GPU_STATS_OUTPUT = "bench_gpu_stats.csv"
BENCH_COLUMNS = {"Benchmark", "BWUtil", "Skipped", GPU_TIME_COLNAME}
BANDWIDTH_COLNAME = "BWUtil"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CV-CUDA benchmarks multiple times and aggregate results."
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        default=1,
        help="Number of times to run each benchmark.",
    )
    parser.add_argument("bench_folder", help="Folder containing benchmark executables.")
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments to pass to the benchmark executables.",
    )
    args = parser.parse_args()

    if args.num_runs <= 0:
        print("E Number of runs must be positive.")
        sys.exit(1)

    bench_args_str = " ".join(args.bench_args)
    bench_folder = args.bench_folder

    try:
        bench_files_all = sorted(os.listdir(bench_folder))
    except FileNotFoundError:
        print(f"E Benchmark folder not found: {bench_folder}")
        sys.exit(1)

    bench_files = [
        fn
        for fn in bench_files_all
        if BENCH_PREFIX in fn and os.path.isfile(os.path.join(bench_folder, fn))
    ]

    if len(bench_files) == 0:
        print(
            f"E No benchmark executables starting with '{BENCH_PREFIX}' found in {bench_folder}"
        )
        sys.exit(1)

    print(f"I Found {len(bench_files)} benchmark executable(s) in {bench_folder}")
    print(f"I Will run each benchmark {args.num_runs} time(s)")
    if bench_args_str:
        print(f"I Passing extra arguments to benchmarks: '{bench_args_str}'")

    l_df_all_runs = []
    all_gpu_times = {}

    for run_num in range(1, args.num_runs + 1):
        print(f"\n--- Starting Run {run_num}/{args.num_runs} ---")
        run_results_found = False
        for filename in bench_files:
            filepath = os.path.join(bench_folder, filename)
            if not os.access(filepath, os.X_OK):
                print(f"W Skipping non-executable file: {filename}")
                continue

            cmd = BENCH_COMMAND.format(filepath, bench_args_str, BENCH_OUTPUT)

            print(f'I Running "{cmd}"', end=" ")
            sys.stdout.flush()

            beg = time.time()
            try:
                process = subprocess.run(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                print(
                    f"\nE Command not found for {filename}. Is it built and in the correct path?"
                )
                continue
            end = time.time()

            print(f"took {end - beg: .3f} sec")

            if process.returncode != 0:
                print(
                    f"W Benchmark exited with error (code {process.returncode}): {filename}"
                )
                print(f"W Stderr: \n{process.stderr}")
                if os.path.exists(BENCH_OUTPUT):
                    os.remove(BENCH_OUTPUT)
                continue

            if not os.path.exists(BENCH_OUTPUT) or os.path.getsize(BENCH_OUTPUT) == 0:
                print(
                    f"W Skipping as benchmark output '{BENCH_OUTPUT}' does not exist "
                    f"or is empty for {filename}"
                )
                continue

            try:
                df = pd.read_csv(BENCH_OUTPUT)
            except pd.errors.EmptyDataError:
                print(
                    f"W Skipping as benchmark output '{BENCH_OUTPUT}' is empty or invalid CSV for {filename}"
                )
                os.remove(BENCH_OUTPUT)
                continue
            except Exception as e:
                print(f"W Error reading CSV '{BENCH_OUTPUT}' for {filename}: {e}")
                os.remove(BENCH_OUTPUT)
                continue

            if not BENCH_COLUMNS.issubset(df.columns):
                missing_cols = BENCH_COLUMNS - set(df.columns)
                print(
                    f"W Skipping {filename} output: Missing required columns: "
                    f"{missing_cols}. Found: {list(df.columns)}"
                )
                os.remove(BENCH_OUTPUT)
                continue

            df_filtered = df[df["Skipped"] == "No"].copy()

            os.remove(BENCH_OUTPUT)

            if len(df_filtered) > 0:
                run_results_found = True
                df_filtered["run"] = run_num
                l_df_all_runs.append(df_filtered)

                # Collect GPU times for statistics
                # Switch to iterrows() for reliable column name access
                for index, row in df_filtered.iterrows():
                    bench_name = getattr(row, BENCH_COLNAME)
                    try:
                        # Access directly using the original column name string
                        gpu_time = float(row[GPU_TIME_COLNAME])
                        all_gpu_times.setdefault(bench_name, []).append(gpu_time)
                    except (KeyError, ValueError, TypeError) as e:
                        print(
                            f"W Could not parse GPU time for '{bench_name}' in run "
                            f"{run_num}. Value: '{row.get(GPU_TIME_COLNAME, 'N/A')}'. Error: {e}"
                        )

            else:
                print(
                    f"W No valid (non-skipped) results found in output for {filename} in run {run_num}"
                )

        if not run_results_found:
            print(
                f"W No benchmark results were successfully processed in run {run_num}."
            )

    print("\n--- Aggregating Results ---")

    if l_df_all_runs:
        df_all = pd.concat(l_df_all_runs, axis=0, ignore_index=True)

        filepath_orig = os.path.join(args.bench_folder, BENCH_RESULTS)
        try:
            df_all.to_csv(filepath_orig, index=False)
            print(
                f"I Full results across {args.num_runs} run(s) written to {filepath_orig}"
            )
        except Exception as e:
            print(f"E Failed to write aggregated results to {filepath_orig}: {e}")

        if BANDWIDTH_COLNAME in df_all.columns:
            try:
                df_all[BANDWIDTH_COLNAME] = pd.to_numeric(
                    df_all[BANDWIDTH_COLNAME], errors="coerce"
                )
                df_summary = (
                    df_all.dropna(subset=[BANDWIDTH_COLNAME])
                    .groupby(BENCH_COLNAME)[BANDWIDTH_COLNAME]
                    .mean()
                )
                if not df_summary.empty:
                    pd.options.display.float_format = "{: .2%}".format
                    print(
                        f"\nI Summary results (Mean {BANDWIDTH_COLNAME} across {args.num_runs} run(s)): "
                    )
                    print(df_summary)
                    pd.options.display.float_format = None
                else:
                    print(
                        f"W Could not compute mean {BANDWIDTH_COLNAME} summary (no valid numeric data)."
                    )
            except Exception as e:
                print(f"W Could not compute mean {BANDWIDTH_COLNAME} summary: {e}")
        else:
            print(f"W Cannot compute {BANDWIDTH_COLNAME} summary: Column not present.")

    else:
        print(
            "W No valid benchmark results collected across any run. Skipping summary generation."
        )

    if all_gpu_times:
        gpu_stats_results = []
        bench_names_with_incomplete_data = []

        for bench_name, times_list in all_gpu_times.items():
            if len(times_list) == args.num_runs:
                try:
                    mean_gpu_time = np.mean(times_list)
                    std_dev_gpu_time = np.std(times_list)
                    gpu_stats_results.append(
                        {
                            BENCH_COLNAME: bench_name,
                            f"Mean {GPU_TIME_COLNAME}": mean_gpu_time,
                            f"Std Dev {GPU_TIME_COLNAME}": std_dev_gpu_time,
                            "Runs": args.num_runs,
                        }
                    )
                except Exception as e:
                    print(f"W Error calculating stats for '{bench_name}': {e}")
            else:
                if len(times_list) > 1:
                    mean_gpu_time = np.mean(times_list)
                    std_dev_gpu_time = np.std(times_list)
                    gpu_stats_results.append(
                        {
                            BENCH_COLNAME: bench_name,
                            f"Mean {GPU_TIME_COLNAME}": mean_gpu_time,
                            f"Std Dev {GPU_TIME_COLNAME}": std_dev_gpu_time,
                            "Runs": len(times_list),
                        }
                    )
                    bench_names_with_incomplete_data.append(
                        f"{bench_name} ({len(times_list)}/{args.num_runs} runs)"
                    )
                elif len(times_list) == 1:
                    mean_gpu_time = times_list[0]
                    std_dev_gpu_time = 0.0
                    gpu_stats_results.append(
                        {
                            BENCH_COLNAME: bench_name,
                            f"Mean {GPU_TIME_COLNAME}": mean_gpu_time,
                            f"Std Dev {GPU_TIME_COLNAME}": std_dev_gpu_time,
                            "Runs": 1,
                        }
                    )
                    bench_names_with_incomplete_data.append(
                        f"{bench_name} (1/{args.num_runs} runs)"
                    )
                else:
                    bench_names_with_incomplete_data.append(
                        f"{bench_name} (0/{args.num_runs} runs)"
                    )

        if bench_names_with_incomplete_data:
            print(
                f"\nW Warning: The following benchmarks had incomplete data "
                f"across runs (results based on available data): "
                f"{', '.join(bench_names_with_incomplete_data)}"
            )

        if gpu_stats_results:
            gpu_stats_df = pd.DataFrame(gpu_stats_results)
            filepath_gpu = os.path.join(args.bench_folder, GPU_STATS_OUTPUT)
            try:
                gpu_stats_df.to_csv(filepath_gpu, index=False, float_format="%.6f")
                print(
                    f"\nI GPU Time statistics across up to {args.num_runs} run(s) written to {filepath_gpu}"
                )
                print(gpu_stats_df.to_string(index=False, float_format="%.6f"))
            except Exception as e:
                print(f"E Failed to write GPU stats results to {filepath_gpu}: {e}")

        else:
            print(
                "W No benchmarks had sufficient GPU time data to calculate statistics."
            )
    else:
        print(
            "W No GPU time data collected across any run. Skipping GPU statistics generation."
        )

    print("\n--- Benchmarking Complete ---")
