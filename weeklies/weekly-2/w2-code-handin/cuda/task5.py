import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the CSV file
df = pd.read_csv('task5.csv')

# Extract columns for the plot
mat_rows = df['mat_rows']
vct_size = df['vct_size']
CPU = df['CPU']
GPU = df['GPU']
SPEEDUP = CPU / GPU

# Create a new figure for the 3D plot
fig = plt.figure(figsize=(12, 8))  # Adjusted figure size for better clarity
ax = fig.add_subplot(111, projection='3d')

# Plot the speedup data
scatter_speedup = ax.scatter(mat_rows, vct_size, SPEEDUP, c='r', label='Speedup', marker='^')

# Label the axes
ax.set_xlabel('Matrix Rows (Log Scale)')
ax.set_ylabel('Vector Size')
ax.set_zlabel('Speedup')
ax.set_title('CPU vs GPU Performance Speedup')

# Add gridlines for better readability
ax.grid(True)

# Add a legend
ax.legend()

# Show the plot
plt.show()