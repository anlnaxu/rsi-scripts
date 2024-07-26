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


def matrix_rank(M):
    return np.linalg.matrix_rank(M)


def count_full_rank_submatrices(matrix, sample_size=10000000):
    full_matrix = np.array(matrix)
    count = 0

    for i in range(sample_size):
        combo = np.random.choice(len(matrix), 27, replace=False)
        submatrix = full_matrix[combo, :]
        if matrix_rank(submatrix) == 27:
            count += 1

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} samples. Current count: {count}")

    return count, sample_size


def main():
    points = label_vector_space_mod3()
    lines = generate_lines(points)
    matrix = create_matrix(lines, len(points))

    print(f"Number of lines: {len(lines)}")
    print("Counting full rank 27x27 submatrices...")
    print(f"Number of lines: {len(lines)}")

    print("\nMatrix representation of lines:")
    for i, row in enumerate(matrix):
        print(f"Line {i + 1}: {row}")

    full_rank_count, samples = count_full_rank_submatrices(matrix)

    print(f"\nNumber of full rank 27x27 submatrices in {samples} random samples: {full_rank_count}")
    print(f"Estimated proportion: {full_rank_count / samples:.6f}")


if __name__ == "__main__":
    main()

