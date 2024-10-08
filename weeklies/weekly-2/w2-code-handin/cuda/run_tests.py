import subprocess
import csv
import re
import sys
import os

# N values for the test
N_VALUES = ["10000", "20000", "30000", "40000", "50000", "60000", "70000", "80000", "90000"]

# Define the GPU test patterns to extract specific GB/sec values
TEST_PATTERNS = [
    r"Naive Reduce with Int32 Addition Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # NRedAdd
    r"Optimized Reduce with Int32 Addition Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # ORedAdd
    r"Naive Reduce with MSSP Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # NRedMssp
    r"Optimized Reduce with MSSP Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # ORedMssp
    r"Scan Inclusive AddI32 GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # Scan
    r"SgmScan Inclusive AddI32 GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)"  # SmgScan
]

# Extract GB/sec values for specific GPU tests
def extract_gb_sec(output):
    gb_sec_values = []
    for pattern in TEST_PATTERNS:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            gb_sec_values.append(float(match.group(1)))
        else:
            gb_sec_values.append(None)  # If the test result is not found, append None
    return gb_sec_values

# Run the test and return the extracted GB/sec values
def run_test(N):
    result = subprocess.run(["./test_pbb", N, "256"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return extract_gb_sec(result.stdout)

# Run the test 3 times for each N, averaging the GB/sec values
def run_and_average(N):
    runs = []
    for _ in range(1):  # Perform 3 iterations
        runs.append(run_test(N))
    # Average the values for each test, ignoring None values
    averages = []
    for test_results in zip(*runs):
        valid_results = [r for r in test_results if r is not None]
        averages.append(sum(valid_results) / len(valid_results) if valid_results else None)
    return averages

# Update the CSV file for the specified column (1, 2, 3, or 4)
def write_to_csv(column_index):
    csv_file = "results.csv"
    # Updated header to match the provided format
    header = [
        "N", "NRedAdd", "NRedAdd_task2", "NRedAdd_task3", "NRedAdd_task2&3",
        "ORedAdd", "ORedAdd_task2", "ORedAdd_task3", "ORedAdd_task3&4",
        "NRedMssp", "NRedMssp_task2", "NRedMssp_task3", "NRedMssp_task2&3",
        "ORedMssp", "ORedMssp_task2", "ORedMssp_task3", "ORedMssp_task2&3",
        "Scan", "Scan_task2", "Scan_task3", "Scan_task2&3",
        "SmgScan", "SmgScan_task2", "SmgScan_task3", "SmgScan_task2&3"
    ]

    # Create the CSV file if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # Read existing CSV file
    existing_data = {}
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            N = row["N"]
            existing_data[N] = row

    # Write updated data
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        for N in N_VALUES:
            avg_results = run_and_average(N)
            if N in existing_data:
                row = existing_data[N]
            else:
                row = {key: "" for key in header}
                row["N"] = N

            # Write results into the appropriate columns based on column_index
            # The pattern repeats every 4 columns for each test
            test_mapping = {
                1: ["NRedAdd", "ORedAdd", "NRedMssp", "ORedMssp", "Scan", "SmgScan"],
                2: ["NRedAdd_task2", "ORedAdd_task2", "NRedMssp_task2", "ORedMssp_task2", "Scan_task2", "SmgScan_task2"],
                3: ["NRedAdd_task3", "ORedAdd_task3", "NRedMssp_task3", "ORedMssp_task3", "Scan_task3", "SmgScan_task3"],
                4: ["NRedAdd_task2&3", "ORedAdd_task3&4", "NRedMssp_task2&3", "ORedMssp_task2&3", "Scan_task2&3", "SmgScan_task2&3"]
            }

            columns = test_mapping[column_index]  # Get the right column set based on the index

            # Ensure that values are written to the correct columns (1st, 2nd, etc., for each test)
            for i, avg in enumerate(avg_results):
                if avg is not None:
                    row[columns[i]] = avg

            writer.writerow(row)

if __name__ == "__main__":
    column_index = int(sys.argv[1])  # Get column index (1, 2, 3, or 4) from command-line argument
    write_to_csv(column_index)