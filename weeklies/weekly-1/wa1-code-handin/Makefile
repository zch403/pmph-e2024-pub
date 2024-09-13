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

# Dynamic rule to run the executable with an argument, excluding common files like Makefile
run_%: $(EXECUTABLE)
	./$(EXECUTABLE) $*

# Clean rule to remove the executable
clean:
	rm -f $(EXECUTABLE)

# Mark default targets as phony to avoid conflicts with filenames
.PHONY: default compile clean run_%