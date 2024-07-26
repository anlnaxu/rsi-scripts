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
# print(generate_lines_and_matrix_mod3())


# Generate the matrix
matrix = generate_lines_and_matrix_mod3()

def remove_last_3_columns(matrix):
    matrix = np.array(matrix)  # Convert to NumPy array if not already
    return matrix[:, :-3]


def remove_zero_rows(matrix):
    matrix = np.array(matrix)
    # Create a boolean mask where True indicates a non-zero row
    mask = np.any(matrix != 0, axis=1)

    # Apply the mask to keep only non-zero rows
    return matrix[mask]

six_matrix = remove_last_3_columns(matrix)

six_matrix = remove_zero_rows(six_matrix)
print(six_matrix)


def generate_submatrices(matrix, k=6):
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    all_combinations = list(combinations(range(rows), k))

    non_full_rank_matrices = []
    count = 0

    for combo in all_combinations:
        submatrix = matrix[list(combo), :]
        if np.linalg.matrix_rank(submatrix) < k:
            non_full_rank_matrices.append(submatrix)
            count += 1

    return non_full_rank_matrices, count


def filter_matrices(matrices):
    # Define the rows to look for
    row1 = np.array([1, 1, 1, 0, 0, 0])
    row2 = np.array([0, 0, 0, 1, 1, 1])

    filtered_matrices = []

    for matrix in matrices:
        # Check if either row1 or row2 is in the matrix
        if any(np.array_equal(row, row1) or np.array_equal(row, row2) for row in matrix):
            filtered_matrices.append(matrix)

    return filtered_matrices


def filter_matrices_with_zero_columns(matrices):
    filtered_matrices = []

    for matrix in matrices:
        # Check if any column in the matrix is all zeros
        if not np.any(np.all(matrix == 0, axis=0)):
            filtered_matrices.append(matrix)

    return filtered_matrices


def filter_matrices_with_specific_column(matrices):
    filtered_matrices = []

    for matrix in matrices:
        match_found = False
        for column in matrix.T:
            if np.sum(column == 1) == 4 and np.sum(column == 0) == 2:
                match_found = True
                break
        if not match_found:
            filtered_matrices.append(matrix)

    return filtered_matrices


def filter_matrices_with_specific_rows(matrices):
    row1 = np.array([1, 1, 1, 0, 0, 0])
    row2 = np.array([0, 0, 0, 1, 1, 1])

    filtered_matrices = []

    for matrix in matrices:
        has_row1 = any(np.array_equal(row, row1) for row in matrix)
        has_row2 = any(np.array_equal(row, row2) for row in matrix)

        # Add matrix to the filtered list if it does not contain both rows
        if not (has_row1 and has_row2):
            filtered_matrices.append(matrix)

    return filtered_matrices


def retain_matrices_with_specific_rows(matrices):
    row1 = np.array([1, 1, 1, 0, 0, 0])
    row2 = np.array([0, 0, 0, 1, 1, 1])

    retained_matrices = []

    for matrix in matrices:
        has_row1 = any(np.array_equal(row, row1) for row in matrix)
        has_row2 = any(np.array_equal(row, row2) for row in matrix)

        # Add matrix to the retained list if it contains both rows
        if has_row1 and has_row2:
            retained_matrices.append(matrix)

    return retained_matrices


submatrices_list, count = generate_submatrices(six_matrix)
filter1_matrices = filter_matrices(submatrices_list)
filter2_matrices = filter_matrices_with_zero_columns(filter1_matrices)
filter3_matrices = filter_matrices_with_specific_column(filter2_matrices)
filter4_matrices = retain_matrices_with_specific_rows(filter3_matrices)

for i, submatrix in enumerate(filter4_matrices, 1):
    print(f"\nSubmatrix {i}:")
    print(submatrix)

def filter_matrices_with_four_ones(matrices_list):
    filtered_matrices = []
    for matrix in matrices_list:
        if any(np.sum(matrix == 1, axis=0) == 4):
            filtered_matrices.append(matrix)
    return filtered_matrices
def filter_matrices_with_four_ones(matrices_list):
    filtered_matrices = []
    for matrix in matrices_list:
        if any(np.sum(matrix == 1, axis=0) == 4):
            filtered_matrices.append(matrix)
    return filtered_matrices


submatrices_list, count = generate_submatrices(six_matrix)

print(f"Number of submatrices without full rank: {count}")

# Filter submatrices with a column containing exactly four 1's
filtered_submatrices = filter_matrices_with_four_ones(submatrices_list)

#print(f"\nNumber of submatrices with a column containing exactly four 1's: {len(filtered_submatrices)}")
#print("\nFiltered submatrices:")
#for i, submatrix in enumerate(filtered_submatrices, 1):
 #   print(f"\nSubmatrix {i}:")
  #  print(submatrix)

def filter_matrices_with_zero_column(matrices_list):
    zero_filtered_matrices = []
    for matrix in matrices_list:
        if np.any(np.all(matrix == 0, axis=0)):
            zero_filtered_matrices.append(matrix)
    return zero_filtered_matrices

zero_submatrices_set = generate_submatrices(six_matrix)

zero_column_matrices = filter_matrices_with_zero_column(zero_submatrices_set)

#print(f"Number of submatrices with at least one column of zeros: {len(zero_column_matrices)}")
#print("\nSubmatrices with at least one column of zeros:")
#for i, matrix_tuple in enumerate(zero_column_matrices, 1):
  #  print(f"\nMatrix {i}:")
  #  print(zero_column_matrices)

