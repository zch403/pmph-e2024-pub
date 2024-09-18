import subprocess
import csv
import re
import sys
import os

# N values for the test
N_VALUES = ["100000000", "50000000", "10000000", "5000000", "1000000", "500000", "100000", "50000", "10000"]

# Define the GPU test patterns to extract specific GB/sec values
TEST_PATTERNS = [
    r"Naive Reduce with Int32 Addition Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # Test1
    r"Optimized Reduce with Int32 Addition Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # Test2
    r"Naive Reduce with MSSP Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # Test3
    r"Optimized Reduce with MSSP Operator.*?Reduce GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # Test4
    r"Scan Inclusive AddI32 GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)",  # Test5
    r"SgmScan Inclusive AddI32 GPU Kernel runs in:.*?GB/sec:\s+(\d+\.\d+)"  # Test6
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
    for _ in range(1):
        runs.append(run_test(N))
    # Average the values for each test, ignoring None values
    averages = []
    for test_results in zip(*runs):
        valid_results = [r for r in test_results if r is not None]
        averages.append(sum(valid_results) / len(valid_results) if valid_results else None)
    return averages

# Write to CSV with test1_1, test1_2, ..., test7_4 structure
def write_to_csv(column_index):
    csv_file = "results.csv"
    header = ["N"] + [f"test{i}_{j}" for i in range(1, 8) for j in range(1, 5)]  # 7 tests, 4 columns each

    # If the file doesn't exist, create it with the header
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # Open the CSV file and write the results
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        for N in N_VALUES:
            avg_results = run_and_average(N)
            # Prepare the row with N and the averaged GB/sec values
            row = [N] + avg_results
            writer.writerow(row)

if __name__ == "__main__":
    column_index = int(sys.argv[1])  # Not using column index directly anymore as the structure is fixed
    write_to_csv(column_index)