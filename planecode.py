import numpy as np
from itertools import combinations
import itertools
def label_vector_space_mod3():
    points = []
    for x in range(3):
        for y in range(3):
            points.append((x, y))

    for i, point in enumerate(points):
        print(f"Point {i + 1}: {point}")

label_vector_space_mod3()

import numpy as np
from itertools import combinations


def generate_lines_and_matrix_mod3():
    points = [(x, y) for x in range(3) for y in range(3)]
    lines = []

    for p1 in points:
        for p2 in points:
            if p1 != p2:
                line = set([p1, p2])
                for p3 in points:
                    if p3 not in line:
                        if (p2[0] - p1[0]) * (p3[1] - p1[1]) % 3 == (p3[0] - p1[0]) * (p2[1] - p1[1]) % 3:
                            line.add(p3)
                if len(line) == 3 and frozenset(line) not in lines:
                    lines.append(frozenset(line))

    matrix = np.zeros((len(lines), len(points)), dtype=int)

    for i, line in enumerate(lines):
        for point in line:
            point_index = points.index(point)
            matrix[i, point_index] = 1

    return matrix  # Make sure to return the matrix
print(generate_lines_and_matrix_mod3())

def find_minimum_rows_for_full_rank(matrix):
    num_rows, num_cols = matrix.shape

    for r in range(1, num_rows + 1):
        for combo in combinations(range(num_rows), r):
            submatrix = matrix[list(combo), :]
            if np.linalg.matrix_rank(submatrix) == 9:
                return r, combo

    return None, None


# Generate the matrix
matrix = generate_lines_and_matrix_mod3()

# Find the minimum number of rows for full rank
min_rows, row_indices = find_minimum_rows_for_full_rank(matrix)

if min_rows is not None:
    print(f"Minimum number of rows needed for full rank: {min_rows}")
    print(f"Indices of these rows: {row_indices}")
else:
    print("No combination of rows achieves full rank.")

counter = 0
for combination in itertools.combinations(matrix, 9):
    submatrix = np.array(combination)
    if np.linalg.matrix_rank(submatrix) == 9:
        counter += 1
        print(submatrix)
print(counter)






