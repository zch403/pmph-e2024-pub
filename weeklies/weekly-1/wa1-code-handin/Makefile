CXX        = nvcc -O3
# -arch=sm_60

SRC = wa1-task3.cu
EXECUTABLE = wa1-task3

# Default target
default: compile

# Compile rule for the CUDA source file
compile: $(EXECUTABLE)

# Compile the CUDA source to an executable
$(EXECUTABLE): $(SRC)
	$(CXX) -o $(EXECUTABLE) $(SRC)

# Rule to run the executable with a numeric argument (like 100)
# This ensures that numeric targets work without a circular dependency
%: $(EXECUTABLE)
	@if [ "$@" -eq "$@" ] 2>/dev/null; then ./$(EXECUTABLE) $@; else echo "Invalid target"; fi

# Clean rule to remove the executable
clean:
	rm -f $(EXECUTABLE)

# Mark default targets as phony to avoid conflicts with filenames
.PHONY: default compile clean