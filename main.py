from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np


def prepare_matrix(full_matrix: np.ndarray, positions: List[int]) -> np.ndarray:
    cur_matrix = deepcopy(full_matrix)
    used = {}
    for pos in positions:
        j = -1
        for i in range(cur_matrix.shape[0]):
            if cur_matrix[i][pos] != 0 and (i not in used):
                j = i
                break
        used[j] = -1
        if j == -1:
            raise Exception("Can't make identitat matrix on the given positions")
        cur_matrix[j] /= cur_matrix[j][pos]
        for i in range(cur_matrix.shape[0]):
            if i != j and abs(cur_matrix[i][pos]) >= 1e-6:
                cur_matrix[i] -= (cur_matrix[j] * cur_matrix[i][pos])
    return cur_matrix


def find_acceptable_solution(full_matrix: np.ndarray) -> Optional[List[int]]:
    cur_matrix = deepcopy(full_matrix)
    m = full_matrix.shape[0]
    n = full_matrix.shape[1]
    f = np.array([0] * (full_matrix.shape[1] - 1) + [-1] * full_matrix.shape[0])
    cur_matrix_list = cur_matrix.tolist()

    for i in range(m):
        additional_line = [0.0] * m
        additional_line[i] = 1.0
        cur_matrix_list[i] += additional_line

    _, positions, val = simplex_method(np.array(cur_matrix_list), f, [i + n for i in range(m)])
    if (abs(val) > 1e-6):
        return None
    return positions


def choose_column(cur_matrix: np.ndarray) -> Optional[int]:
    n = cur_matrix.shape[1] - 1
    for i in range(1, n + 1):
        if cur_matrix[-1][i] >= 1e-6:
            print(f"Chosen column is {i}")
            return i
    return None


def choose_row(cur_matrix: np.ndarray, chosen_column: int) -> Optional[int]:
    # todo: check if function has maximum
    result = 1e9
    result_ind = -1
    b = cur_matrix[:, 0]
    column = cur_matrix[:, chosen_column]
    print(b)
    print(column)
    for i in range(len(b) - 1):
        if column[i] > 1e-6 and b[i] / column[i] < result:
            result_ind = i
            result = b[i] / column[i]
    # print(result_ind)
    return result_ind


def simplex_method(full_matrix: np.ndarray, f: np.ndarray, positions: List[int]) -> Optional[
    Tuple[np.ndarray, List[int], float]]:
    cur_matrix = deepcopy(full_matrix)
    m = cur_matrix.shape[0]
    cur_matrix = np.vstack((cur_matrix, np.append(np.array([0]), f)))
    # print(np.append(np.array([0]), f))
    poss = [-1 for _ in range(len(positions))]
    # print(positions)
    for i in range(m):
        for j in range(len(positions)):
            if cur_matrix[i][positions[j]] >= 0.5:
                poss[j] = i
    # print(cur_matrix)
    # print("poss: ", poss)
    for j in range(len(positions)):
        cur_matrix[-1] -= cur_matrix[-1][positions[j]] * cur_matrix[poss[j]]

    it = 0
    print(cur_matrix)
    while True and it < 100:
        it += 1
        chosen_column = choose_column(cur_matrix)
        if chosen_column is None:
            break
        chosen_row = choose_row(cur_matrix, chosen_column)
        cur_matrix[chosen_row] /= cur_matrix[chosen_row][chosen_column]
        for i in range(cur_matrix.shape[0]):
            if i != chosen_row and abs(cur_matrix[i][chosen_column]) >= 1e-6:
                cur_matrix[i] -= cur_matrix[chosen_row] * cur_matrix[i][chosen_column]
        # print(chosen_row, chosen_column)
        # exit(0)
        positions[chosen_row] = chosen_column
    pre_result = np.zeros(cur_matrix.shape[1] - 1)
    for i in range(len(positions)):
        pre_result[positions[i] - 1] = cur_matrix[i][0]
    return pre_result, positions, -cur_matrix[-1][0]


def solve(full_matrix: np.ndarray, f: np.ndarray, positions: Optional[List[int]]):
    if positions is None:
        return None
    prepared_matrix = prepare_matrix(full_matrix, positions)
    # print("Prep_matr:\n", prepared_matrix)
    opt_sol, pos, val = simplex_method(prepared_matrix, f, positions)
    print("Optimal solution: ", opt_sol)
    print("Value of function: ", val)


if __name__ == '__main__':
    input = [[1, 2, 3], [4, 5, 6]]
    x0 = [1, 0, 0]
    b = [1, 2, 3]
    f = [4, 5, 6]

    full_matrix = np.array([[2.0, 1.0, -1.0, 10.0],
                            [4.0, 1.0, 1.0, 1.0]])
    positions = find_acceptable_solution(full_matrix)
    solve(full_matrix, np.array(f), positions)

    print("hello")