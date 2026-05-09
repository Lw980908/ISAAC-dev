
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."


# Python Operator Performance Benchmarking

Using various performance benchmarking scripts that ships with CV-CUDA samples, we can measure and report the performance of various CV-CUDA operators from Python.

The following scripts are part of the performance benchmarking tools in CV-CUDA.

1. `bench/python/benchmark.py`
2. `bench/python/perf_utils.py`
3. `bench/python/bench_utils.py`

We use NVIDIA NSYS to profile the code for its CPU and GPU run-times. Profiling is done at the operator level using NVTX style markers to annotate the code using `push_range()` and `pop_range()` methods, then benchmarking is run to collect the timing information of all such ranges. Please refer to the NVIDIA NSIGHT user guide to learn more about NSIGHT and NVTX trace (https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## Installation

### Prerequisites

- **CUDA Toolkit**: CUDA 12.2+ or CUDA 13.0+
- **Python**: 3.10 - 3.13 (3.10+ recommended for benchmarking)
- **System**: Ubuntu 22.04+ or compatible Linux distribution

### System Dependencies

Run the installation script to install system dependencies, NVIDIA Nsight Systems, and Python packages:

```shell
cd bench/python
./install_dependencies.sh
```

**Note:** The script will prompt for your sudo password when installing system packages.

This script will:
- Detect your CUDA version (supports CUDA 12 and CUDA 13)
- Install NVIDIA Nsight Systems 2025.5.1 (required for profiling)
- Install ffmpeg and other media processing libraries
- Set up build tools and dependencies
- **Create a Python virtual environment and install all required Python packages**

**Note**: If you already have NSYS installed, the script will skip the installation. Minimum required version: NSYS 2025.5.1.

After installation, activate the virtual environment:
```shell
source venv_bench/bin/activate
```

### Python Environment

The `install_dependencies.sh` script automatically sets up a Python virtual environment and installs all required dependencies from self-contained requirements files in the `bench/python/` directory.

**Manual Installation (if needed):**

If you need to reinstall Python packages or set up on a system where you've already installed system dependencies:

```shell
cd bench/python

# Ensure you have system dependencies installed
# If not, run: sudo ./install_dependencies.sh

# Activate the virtual environment (created by install_dependencies.sh)
source venv_bench/bin/activate

# Reinstall Python packages for your CUDA version
# For CUDA 12:
python3 -m pip install -r requirements_cu12.txt

# For CUDA 13:
python3 -m pip install -r requirements_cu13.txt
```

**Note:** All Python dependencies are self-contained in three requirements files:
- `requirements_common.txt`: Common dependencies (NumPy, benchmarking tools)
- `requirements_cu12.txt`: CUDA 12-specific packages (includes common requirements)
- `requirements_cu13.txt`: CUDA 13-specific packages (includes common requirements)

## benchmark.py

This is the main launcher script responsible for launching operator benchmarks. After launching it:

1. Coordinates the entire process of benchmarking with NSYS.
2. Parses various results returned by NSYS.
3. Stores per-process results in a JSON file.
4. Automatically calculates average numbers across all processes and also saves them in a JSON file.

## perf_utils.py

This file holds the data structures and functions most commonly used during the benchmarking process:

1. It contains the `CvCudaPerf` class to hold performance data with an API similar to NVTX.
2. Provides a way to maximize the GPU clocks before benchmarking.
3. Provides a command-line argument parser that can be shared across all benchmarks to maintain uniformity in the way of passing the inputs.

The `CvCudaPerf` class is used to mark the portions of code that one wants to benchmark using its `push_range()` and `pop_range()` methods. These methods are similar to NVTX except that it adds two new features on top that allows us to compute more detailed numbers:
1. It can record the start of a batch. A batch is a logical group of operations that often repeats.
2. It can record the end of a batch, with its batch size.

## About the Operator Benchmarks

Operators for which a test case has been implemented in the `all_ops` folder can be benchmarked. The following statements are true for all such test cases:

1. All inherit from a base class called `AbstractOpBase` which allows them to expose benchmarking capabilities in a consistent manner. They all have a setup stage, a run stage and an optional visualization stage. By default, the visualization is turned off.
2. All receive the same input image. Some operators may need to read additional data. Such data is always read from the `assets` directory.
3. All run for a number of iterations (default is set to 10) and a batch size (default is set to 32).
4. The script `benchmark.py` handles overall benchmarking. It launches the runs, monitors it, communicates with NSYS and saves the results of a run in a JSON file. Various settings such as using warm-up (default is set to 1 iteration) are handled here.
5. One or more benchmark runs can be compared and summarized in a table showing only the important information from the detailed JSON files.

## Setting up the environment

1. Follow [Setting up the environment](../../samples/README.md#setting-up-the-environment) section of the CV-CUDA samples. Note: The step asking to install dependencies can be ignored if you are only interested in benchmarking the operators (and not the samples).


## Running the benchmark

The script `run_bench.py` together with `benchmark.py` can be used to automatically benchmark all supported CV-CUDA operators in Python. Additionally, one or more runs can be summarized and compared in a table using the functionality provided by `bench_utils.py`


### To run the operator benchmarks

```bash
python3 bench/python/benchmark.py -o <OUT_DIR> bench/python/run_bench.py
```
- Where:
    1. An `OUTPUT_DIR` must be given to store various benchmark artifacts.
- Upon running it will:
    1. Ask the `benchmark.py` to launch the `run_bench.py`.
    2. `run_bench.py` will then find out all the operators that can be benchmarked.
    3. Run those one by one, through all the stages, such as setup, run and visualization (if enabled).
    4. Store the artifacts in the output folder. This is where the `benchmark.py` style `benchmark_mean.json` would be stored.

Once a run is completed, one can use the `bench_utils.py` to summarize it. Additionally, we can use the same script to compare multiple different runs.

### To summarize one run only

```bash
python3 bench/python/bench_utils.py -o <OUTPUT_DIR> -b <benchmark_mean_json_path> -bn baseline
```
- Where:
    1. A `OUTPUT_DIR` must be given to store the summary table as a CSV file.
    2. The first run's `benchmark_mean.json` path must be given as `b`.
    3. The display name of the first run must be given as `bn`.
- Upon running it will:
    1. Grab appropriate values from the JSON file for all the operators and put it in a table format.
    2. Save the table as a CSV file.

The output CSV file will be stored in the `OUTPUT_DIR` with current date and time on it.

NOTE: `benchmark.py` will produce additional JSON files (and visualization files if it was enabled). These files provide way more detailed information compared to the CSV and is usually only meant for debugging purposes.


### To summarize and compare multiple runs

```bash
python3 bench/python/bench_utils.py -o <OUTPUT_DIR> -b <benchmark_mean_json_path> -bn baseline \
       -c <benchmark_mean_2_json_path> -cn run_2 \
       -c <benchmark_mean_3_json_path> -cn run_3
```
- Where:
    1. An `OUTPUT_DIR` must be given to store the summary table as a CSV file.
    2. The first run's `benchmark_mean.json` path is given as `b`.
    3. The display name of the first run is given as `bn`.
    4. The second run's `benchmark_mean.json` path is given as `c`.
    5. The display name of the second run is given as `cn`.
    6. The third run's `benchmark_mean.json` path is given as `c`.
    7. The display name of the third run must be given as `cn`.
    8. Options `c` and `cn` can be repeated as zero or more times to cover all the runs.
- Upon running it will:
    1. Grab appropriate values from the JSON file for all the operators and put it in a table format.
    2. Save the table as a CSV file.


## Interpreting the results

Upon a successful completion of the benchmarking process, we get several files with performance data:

### Understanding the benchmark JSON files

The `benchmark.py` script produces the following files:

1. **Per process statistics** in `benchmark.json` files. These are stored in `<OUTPUT_DIR>/proc_X_gpu_Y` where `OUTPUT_DIR` is the directory used to store the output, `X` is the CPU index and `Y` is the GPU index.

    In each `benchmark.json` file one can see an overall structure like this:
    ```json
    {
        "data": {},
        "mean_data": {},
        "meta": {}
    }
    ```

    - The `data` key stores the per batch data maintaining the hierarchy of the pipeline. At each non-batch level, it stores the following information:
        ```json
        "pipeline": {
            "batch_0": {
                "stage_1": {
                    "cpu_time": 300.797,
                    "gpu_time": 15.2345
                },
                "stage_2": {
                    "cpu_time": 300.416,
                    "gpu_time": 28.5678
                },
                "cpu_time": 601.373,
                "gpu_time": 43.8023,
                "total_items": 1,
                "cpu_time_per_item": 601.373,
                "gpu_time_per_item": 43.8023
            }
        }
        ```
        One can see the `cpu_time` and `gpu_time` per stage (in milliseconds). The `total_items` (i.e. the batch size) is also tracked at the batch level and per item numbers are computed from it.

    - At the batch level, the statistics are aggregated from all the batches and reported considering the warm-up batches. The `*_minus_warmup` timings are the ones which ignore the warm-up batches from the computation:
        ```json
        {
            "cpu_time": 1805.027,
            "gpu_time": 131.4069,
            "cpu_time_per_item": 601.676,
            "gpu_time_per_item": 43.8023,
            "total_items": 3,
            "cpu_time_minus_warmup": 601.759,
            "gpu_time_minus_warmup": 87.6046,
            "cpu_time_per_item_minus_warmup": 601.759,
            "gpu_time_per_item_minus_warmup": 43.8023,
            "total_items_minus_warmup": 2
        }
        ```

    - The `mean_data` key stores the average of all numbers across all the batches.
    - The `meta` key stores various metadata about the run. This may be useful for reproducibility purposes.

2. **Overall statistics** in the `benchmark_mean.json` file. This file will be stored in `<OUTPUT_DIR>`.

    In `benchmark_mean.json` file one can see an overall structure like this:
    ```json
    {
        "mean_all_batches": {},
        "mean_data": {},
        "meta": {}
    }
    ```

    - The `mean_all_batches` key stores average per batch numbers from all processes launched by the `benchmark.py`. These are essentially the mean of the `data` field reported in the per process' `benchmark.json` file and maintains the overall pipeline hierarchy.
    - The `mean_data` key stores the average numbers from all batches from all processes. These are essentially the mean of the `mean_data` reported in the per process' `benchmark.json` file.
    - The `meta` key stores various metadata about the run. This may be useful for reproducibility purposes.

**NOTE**: `benchmark.py` will produce additional JSON files (and visualization files if it was enabled). These files provide way more detailed information compared to the CSV and is usually only meant for debugging purposes.

### Understanding the summary CSV files

Upon a successful completion of the `bench_utils.py` script, we get a CSV file that summarizes the benchmark results.

- If you ran it only on one run, your CSV will only have four columns - showing data only from that run:
    1. `index`: from 0 to N-1 for all the N operators benchmarked
    2. `operator name` The name of the operator
    3. `baseline run time (ms)`: The first run's time in milliseconds, averaged across M iterations (default is 10, with warm-up runs discarded)
    4. `run time params`: Any helpful parameters supplied to the operator as it ran in first run. Only lists primitive data-types.

- If you ran it on more than one runs, your CSV file will have additional columns - comparing data of those runs with the baseline run. Additional columns, per run, would be:
    1. `run i time (ms)`: The ith run's time in milliseconds, averaged across M iterations (default is 10, with warm-up runs discarded)
    2. `run i v/s baseline speed-up`: The speed-up factor. This is calculated by dividing `run i time (ms)` by `baseline run time (ms)`.

## Regarding maximizing the clocks

Often during the GPU benchmarking process one would like to set the GPU clocks and power to their maximum settings. While the `nvidia-smi` command and `nvml` APIs both provide various options to do so, we have consolidated these into a convenient function call `maximize_clocks()` in the `perf_utils.py` script. One can easily turn it on during the benchmarking process by passing the `--maximize_clocks` flag to the `benchmark.py` script. This will also bring the clocks down to its original values once the process is over.
