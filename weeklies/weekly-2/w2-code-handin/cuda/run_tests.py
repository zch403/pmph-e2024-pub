import subprocess
import csv
import re
import sys
import os

# N values for the test
N_VALUES = ["100000000", "50000000", "10000000", "5000000", "1000000", "500000", "100000", "50000", "10000"]

# Extract GB/sec from the test output using regex
def extract_gb_sec(output):
    return [float(match) for match in re.findall(r'GB/sec:\s+(\d+\.\d+)', output)]

# Run the test and return the extracted GB/sec values
def run_test(N):
    result = subprocess.run(["./test_pbb", N, "256"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return extract_gb_sec(result.stdout)

# Run the test 5 times for each N, averaging the GB/sec values
def run_and_average(N):
    runs = []
    for _ in range(5):
        runs.append(run_test(N))
    return [sum(x)/len(x) for x in zip(*runs)]

# Write to CSV with an argument specifying which column to write to
def write_to_csv(column_index):
    csv_file = "results.csv"
    header = ["N"] + [f"GB/sec_{i+1}" for i in range(4)]  # 4 GB/sec columns

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
            row = [N] + [""] * 4  # Initialize the row with empty GB/sec columns
            row[column_index] = avg_results[column_index - 1]  # Fill the specific GB/sec column
            writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./run_tests.py <column_index>")
        sys.exit(1)

    column_index = int(sys.argv[1])  # The column index (1 to 4) to write the GB/sec values to
    if column_index < 1 or column_index > 4:
        print("Invalid column index. Please choose a value between 1 and 4.")
        sys.exit(1)

    write_to_csv(column_index)