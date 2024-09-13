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

# Dynamic rule to run the executable with an argument
# The target will be any number (like 100), and $@ will represent the argument
%: $(EXECUTABLE)
	./$(EXECUTABLE) $@

# Clean rule to remove the executable
clean:
	rm -f $(EXECUTABLE)

# Mark default targets as phony to avoid issues with files named the same as targets
.PHONY: default compile clean