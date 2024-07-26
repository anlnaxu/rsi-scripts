import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Points in the 3D vector space mod 3
points = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]

# Given result matrix
matrix = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
])

# Function to plot lines
def plot_lines(matrix, points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Loop through each row in the matrix
    for i, row in enumerate(matrix):
        line_points = [points[j] for j, val in enumerate(row) if val == 1]
        x_vals = [p[0] for p in line_points]
        y_vals = [p[1] for p in line_points]
        z_vals = [p[2] for p in line_points]
        
        ax.plot(x_vals, y_vals, z_vals, color=colors[i % len(colors)], label=f"Line {i+1}")

        # Annotate points
        for (x, y, z) in line_points:
            ax.text(x, y, z, f"({x},{y},{z})")

    # Setting the axes properties
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set axis limits
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([0, 2])

    # Adding a legend
    ax.legend()

    plt.show()

# Plot the lines
plot_lines(matrix, points)