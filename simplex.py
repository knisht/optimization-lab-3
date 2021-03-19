from copy import deepcopy
from typing import Optional, List, Tuple

import numpy as np

from utils import format_list


def find_acceptable_solution(full_matrix: List[List[float]]) -> Optional[Tuple[List[int], List[float]]]:
    full_matrix_np = np.array(full_matrix)
    cur_matrix = deepcopy(full_matrix_np)
    m = full_matrix_np.shape[0]
    n = full_matrix_np.shape[1]
    f = np.array([0] * (full_matrix_np.shape[1] - 1) + [-1] * full_matrix_np.shape[0])
    cur_matrix_list = cur_matrix.tolist()
    for i in range(m):
        additional_line = [0.0] * m
        additional_line[i] = 1.0
        cur_matrix_list[i] += additional_line

    opt, positions, val = simplex_method(np.array(cur_matrix_list), f, [i + n for i in range(m)])
    if abs(val) > 1e-6:
        return None
    res = opt.tolist()
    return positions, res[:-m]


def choose_column(cur_matrix: np.ndarray) -> Optional[int]:
    n = cur_matrix.shape[1] - 1
    for i in range(1, n + 1):
        if cur_matrix[-1][i] >= 1e-6:
            # print(f"Chosen column is {i}")
            return i
    return None


def choose_row(cur_matrix: np.ndarray, chosen_column: int, positions: List[int]) -> Optional[int]:
    # todo: check if function has maximum
    result = 1e9
    result_ind = -1
    b = cur_matrix[:, 0]
    column = cur_matrix[:, chosen_column]
    # print(b)
    # print(column)
    # print(chosen_column)
    for i in range(len(b) - 1):
        # print(b[i], column[i])
        if column[i] > 1e-6 and abs(b[i] / column[i] - result) < 1e-6 and positions[result_ind] > positions[i]:
            result_ind = i
            result = b[i] / column[i]
            continue
        if column[i] > 1e-6 and b[i] / column[i] < result:
            result_ind = i
            result = b[i] / column[i]
    if result_ind == -1:
        return None
    return result_ind


def simplex_method(full_matrix: np.ndarray, f: np.ndarray, positions: List[int]) -> \
        Optional[Tuple[np.ndarray, List[int], float]]:
    cur_matrix = deepcopy(full_matrix)
    m = cur_matrix.shape[0]
    cur_matrix = np.vstack((cur_matrix, np.append(np.array([0]), deepcopy(f))))
    sorted_poss = []
    for i in range(m):
        for j in range(len(positions)):
            if cur_matrix[i][positions[j]] >= 0.5:
                sorted_poss.append(positions[j])
    for j in range(len(positions)):
        cur_matrix[-1] -= cur_matrix[-1][sorted_poss[j]] * cur_matrix[j]
    it = 0
    while True and it < 100:
        it += 1
        # my_res = np.zeros(cur_matrix.shape[1] - 1)
        # val = 0.0
        # for i in range(len(sorted_poss)):
        #     my_res[sorted_poss[i] - 1] = cur_matrix[i][0]
        #     val += cur_matrix[i][0] * f[sorted_poss[i] - 1]
        # print(format_list(my_res.tolist()))
        # print(cur_matrix[-1][0], )
        chosen_column = choose_column(cur_matrix)
        if chosen_column is None:
            break
        chosen_row = choose_row(cur_matrix, chosen_column, positions)
        if chosen_row is None:
            return chosen_row
        cur_matrix[chosen_row] /= cur_matrix[chosen_row][chosen_column]
        for i in range(cur_matrix.shape[0]):
            if i != chosen_row and abs(cur_matrix[i][chosen_column]) >= 1e-6:
                cur_matrix[i] -= cur_matrix[chosen_row] * cur_matrix[i][chosen_column]
        sorted_poss[chosen_row] = chosen_column
        # print(sorted_poss)
    if it == 100:
        raise Exception("Iteration limit")
    pre_result = np.zeros(cur_matrix.shape[1] - 1)
    for i in range(len(sorted_poss)):
        pre_result[sorted_poss[i] - 1] = cur_matrix[i][0]
    # print("----")
    return pre_result, sorted_poss, -cur_matrix[-1][0]

