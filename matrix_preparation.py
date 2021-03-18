from copy import deepcopy
from typing import List

import numpy as np


def prepare_matrix(full_matrix: List[List[float]], positions: List[int]) -> np.ndarray:
    cur_matrix = deepcopy(np.array(full_matrix))
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
