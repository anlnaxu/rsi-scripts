import itertools
import numpy as np

def label_vector_space_mod3():
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                points.append((x, y, z))
    return points

def is_collinear(p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    det1 = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    det2 = (x2 - x1) * (z3 - z1) - (x3 - x1) * (z2 - z1)
    det3 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    return (det1 % 3 == 0) and (det2 % 3 == 0) and (det3 % 3 == 0)

def generate_lines(points):
    lines = []
    for combo in itertools.combinations(range(len(points)), 3):
        if is_collinear(points[combo[0]], points[combo[1]], points[combo[2]]):
            line = set()
            for i in range(len(points)):
                if is_collinear(points[combo[0]], points[combo[1]], points[i]):
                    line.add(i)
            lines.append(line)
    return list(set(frozenset(line) for line in lines))

def create_matrix(lines, num_points):
    matrix = []
    for line in lines:
        row = [1 if i in line else 0 for i in range(num_points)]
        matrix.append(row)
    return matrix

def get_z0_submatrix(full_matrix, points):
    z0_lines = []
    for i, line in enumerate(full_matrix):
        if sum(line[j] for j, p in enumerate(points) if p[2] == 0) >= 3:
            z0_lines.append(line)

    z0_matrix = np.array(z0_lines)  # No transpose needed

    # Select 9 linearly independent rows using QR decomposition
    q, r = np.linalg.qr(z0_matrix.T)
    rank = np.linalg.matrix_rank(r)
    selected_rows = np.where(np.abs(r.diagonal()) > 1e-10)[0][:9]

    return z0_matrix[selected_rows, :]

# Generate points and full matrix
points = label_vector_space_mod3()
lines = generate_lines(points)
full_matrix = create_matrix(lines, len(points))

# Get z=0 submatrix
z0_submatrix = get_z0_submatrix(full_matrix, points)

#print("Z=0 submatrix shape:", z0_submatrix.shape)
#print("Z=0 submatrix rank:", np.linalg.matrix_rank(z0_submatrix))
#print(z0_submatrix)

def get_x2_submatrix(full_matrix, points):
    x2_lines = []
    for i, line in enumerate(full_matrix):
        if sum(line[j] for j, p in enumerate(points) if p[0] == 2) >= 3:
            x2_lines.append(line)

    x2_matrix = np.array(x2_lines)  # No transpose needed

    # Select 9 linearly independent rows using QR decomposition
    q, r = np.linalg.qr(x2_matrix.T)
    rank = np.linalg.matrix_rank(r)
    selected_rows = np.where(np.abs(r.diagonal()) > 1e-10)[0][:9]

    return x2_matrix[selected_rows, :]

# Generate points and full matrix
points = label_vector_space_mod3()
lines = generate_lines(points)
full_matrix = create_matrix(lines, len(points))

# Get x=2 submatrix
x2_submatrix = get_x2_submatrix(full_matrix, points)

#print("x=2 submatrix shape:", x2_submatrix.shape)
#print("x=2 submatrix rank:", np.linalg.matrix_rank(x2_submatrix))
#print(x2_submatrix)

combined_matrix = np.vstack((z0_submatrix, x2_submatrix))
combined_matrix = np.unique(combined_matrix, axis=0)

# Calculate the rank of the combined matrix
combined_matrix_rank = np.linalg.matrix_rank(combined_matrix)

print("Combined matrix shape:", combined_matrix.shape)
print("Combined matrix rank:", combined_matrix_rank)
print(combined_matrix)

counter = 0
total_combinations = 0

# Iterate over all combinations of 15 rows from the combined matrix
for combination in itertools.combinations(x2_submatrix, 6):
    combined_matrix_final = np.vstack((z0_submatrix, combination))
    combined_matrix_final = np.unique(combined_matrix_final, axis=0)
    submatrix = np.array(combined_matrix_final)
    if np.linalg.matrix_rank(submatrix) == 15:
        counter += 1
        print(submatrix)
print(counter)




