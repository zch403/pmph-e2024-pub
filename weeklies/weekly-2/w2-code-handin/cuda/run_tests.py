import csv
import subprocess
import os

N_VALUES = ["100000000", "50000000", "10000000", "5000000", "1000000", "500000", "100000", "50000", "10000"]

# Define new column headers
column_headers = [
    "N", "NRedAdd", "NRedAdd_task2", "NRedAdd_task3", "NRedAdd_task2&3",
    "ORedAdd", "ORedAdd_task2", "ORedAdd_task3", "ORedAdd_task3&4",
    "NRedMssp", "NRedMssp_task2", "NRedMssp_task3", "NRedMssp_task2&3",
    "ORedMssp", "ORedMssp_task2", "ORedMssp_task3", "ORedMssp_task2&3",
    "Scan", "Scan_task2", "Scan_task3", "Scan_task2&3",
    "SmgScan", "SmgScan_task2", "SmgScan_task3", "SmgScan_task2&3"
]

def run_test(N):
    result = subprocess.run(["./test_pbb", N, "256"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    return output

def extract_gb_per_sec(output):
    # Extract the GB/sec values for the relevant tests
    gb_per_sec = []
    for line in output.splitlines():
        if "GPU Kernel runs in" in line and "GB/sec:" in line:
            gb_per_sec.append(float(line.split("GB/sec:")[1].strip()))
    return gb_per_sec

def run_and_average(N):
    runs = []
    for _ in range(1):  # Perform 3 iterations
        output = run_test(N)
        runs.append(extract_gb_per_sec(output))
    
    # Calculate the average for each test across 3 runs
    avg_results = [sum(x) / len(x) for x in zip(*runs)]
    return avg_results

def write_to_csv(column_index):
    # Check if file exists
    file_exists = os.path.isfile("results.csv")
    
    # Open file for appending or writing
    with open("results.csv", mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header if file doesn't exist
        if not file_exists:
            writer.writerow(column_headers)
        
        # For each value of N
        for N in N_VALUES:
            avg_results = run_and_average(N)
            
            # Prepare the row, filling in empty columns except for the column index being written
            row = [N] + [""] * (len(column_headers) - 1)
            
            # Fill in the correct column with the averaged results for this N
            row[1 + (column_index - 1) * 4 : 1 + column_index * 4] = avg_results
            
            writer.writerow(row)

if __name__ == "__main__":
    import sys
    column_index = int(sys.argv[1])
    write_to_csv(column_index)