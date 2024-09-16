#!/bin/bash
# Script to run test_pbb with different values of N, extract running time and throughput

# Define the values for N
N_VALUES=("100003565" "10000000"  "1000000"   "100000")
CUDA_BLOCK_SIZE=256

# Create or clear the output file
OUTPUT_FILE="test_pbb_results.txt"
echo "" > $OUTPUT_FILE

# Loop over the different values of N
for N in "${N_VALUES[@]}"; do
    # Run the test and capture the output
    ./test_pbb $N $CUDA_BLOCK_SIZE > temp_output.txt

    # Extract running time and throughput
    echo "Results for N = $N" >> $OUTPUT_FILE
    
    grep "Naive Memcpy GPU Kernel runs in:" temp_output.txt >> $OUTPUT_FILE
    grep "Reduce GPU Kernel runs in:" temp_output.txt | head -n 1 >> $OUTPUT_FILE
    grep "Reduce GPU Kernel runs in:" temp_output.txt | head -n 2 | tail -n 1 >> $OUTPUT_FILE
    grep "Reduce GPU Kernel runs in:" temp_output.txt | tail -n 2 | head -n 1 >> $OUTPUT_FILE
    grep "Scan Inclusive AddI32 GPU Kernel runs in:" temp_output.txt >> $OUTPUT_FILE
    grep "SgmScan Inclusive AddI32 GPU Kernel runs in:" temp_output.txt >> $OUTPUT_FILE

    echo "" >> $OUTPUT_FILE
done

# Clean up
rm temp_output.txt

echo "Test results have been saved to $OUTPUT_FILE"