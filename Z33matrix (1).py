import numpy as np

# Define the mapping of labels (A to AA) to their corresponding triplets (x, y, z)

def generate_label_to_vertex():
    label_to_vertex = {}
    label = 1
    for z in range(3):
        for y in range(3):
            for x in range(3):
                label_to_vertex[label] = (x, y, z)
                label += 1
    return label_to_vertex
    
label_to_vertex = generate_label_to_vertex()

# Generate all unique lines passing through 3 vertices in the 3D space modulo 3
def generate_lines(vertices):
    lines = []
    n = len(vertices)
    
    # Iterate through all combinations of vertices
    for i in range(n):
        for j in range(i + 1, n):
            v1 = vertices[i]
            v2 = vertices[j]
            
            # Find a third vertex that is collinear with v1 and v2 modulo 3
            for k in range(n):
                v3 = vertices[k]
                if v1 != v2 and v2!=v3 and v3!=v1 and is_collinear(v1, v2, v3):
                    line = tuple(sorted([v1, v2, v3]))
                    if line not in lines:
                        lines.append(line)
    
    return lines

# Function to check if three vertices are collinear modulo 3
def is_collinear(v1, v2, v3):
    # Vectors from v1 to v2 and v1 to v3
    v1_v2 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    v1_v3 = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
    
    # Cross product of v1_v2 and v1_v3
    cross_product = (
        v1_v2[1] * v1_v3[2] - v1_v2[2] * v1_v3[1],
        v1_v2[2] * v1_v3[0] - v1_v2[0] * v1_v3[2],
        v1_v2[0] * v1_v3[1] - v1_v2[1] * v1_v3[0]
    )
    
    # Check if the cross product is (0, 0, 0), meaning the vectors are collinear
    return cross_product == (0, 0, 0)

# Generate all lines passing through 3 vertices in the 3D space modulo 3
lines = generate_lines(list(label_to_vertex.values()))

# Create a matrix to store vertex labels (1 if vertex is part of a collinear line, 0 otherwise)
num_vertices = len(label_to_vertex)
num_lines = len(lines)
matrix = np.zeros((num_vertices, num_lines), dtype=int)

# Populate the matrix with labels
for vertex_label, vertex_coords in label_to_vertex.items():
    for line_index, line in enumerate(lines):
        if vertex_coords in line:
            matrix[vertex_label - 1, line_index] = 1

# Print the matrix
print(matrix)

Z33 = np.transpose(matrix)

rank = np.linalg.matrix_rank(Z33)
print(f"The rank of the matrix is: {rank}")

rows, cols = Z33.shape
print(f"The matrix has {rows} rows and {cols} columns.")

rank = np.linalg.matrix_rank(matrix)
print(f"The rank of the matrix is: {rank}")
