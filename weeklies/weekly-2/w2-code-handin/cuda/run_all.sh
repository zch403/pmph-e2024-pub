#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <N>"
  exit 1
fi

# Assign the argument to N
N=$1

# Run the commands
echo "Running git pull..."
git pull

echo "Compiling with make..."
make compile

echo "Running tests with python3 for column $N..."
python3 run_tests.py $N

echo "Script completed successfully."