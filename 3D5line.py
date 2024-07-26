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
    return list(frozenset(line) for line in lines)

def create_matrix(lines, num_points):
    matrix = []
    for line in lines:
        row = [1 if i in line else 0 for i in range(num_points)]
        matrix.append(row)
    return matrix

# Generate points and full matrix
points = label_vector_space_mod3()
lines = generate_lines(points)
full_matrix = create_matrix(lines, len(points))

def print_matrix_with_dimensions(matrix):
    # Print dimensions
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    print(f"Matrix dimensions: {rows}x{cols}")

    # Print matrix
    print("Matrix:")
    for row in matrix:
        print(row)

print_matrix_with_dimensions(full_matrix)


def check_all_subsets_rank(matrix):
    num_lines = len(matrix)
    subset_size = 5
    all_subsets = itertools.combinations(range(num_lines), subset_size)

    subset_idx = 1
    found_results = 0
    max_results = 10
    for subset_indices in all_subsets:
        subset_matrix = [matrix[i] for i in subset_indices]

        # Convert subset_matrix to numpy array for rank calculation
        subset_matrix_np = np.array(subset_matrix)
        rank = np.linalg.matrix_rank(subset_matrix_np)

        if rank < 5:
            print(f"Result {found_results + 1}:")
            print(f"Subset {subset_idx}: Rank {rank}")
            print_matrix_with_dimensions(subset_matrix)
            print("\n")
            found_results += 1
            if found_results >= max_results:
                break

        subset_idx += 1

# Check all subsets
check_all_subsets_rank(full_matrix)