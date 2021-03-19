from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from solution import full_solve_maximize


def solve_maximize_leq(full_matrix: List[List[float]], f: List[float], isIneq: List[bool]) \
        -> Optional[Tuple[np.ndarray, float, List[int], List[float]]]:
    cur_matrix = deepcopy(full_matrix)
    n = len(full_matrix[0])
    for i in range(len(full_matrix)):
        if isIneq[i]:
            f.append(0)
            for j in range(len(full_matrix)):
                if i != j:
                    cur_matrix[j].append(0.0)
                else:
                    cur_matrix[j].append(1.0)
    result = full_solve_maximize(cur_matrix, f)
    if result is None:
        return None
    opt, val, pos, vec = result
    return opt[:n], val, pos, vec


def solve_minimize_leq(full_matrix: List[List[float]], f: List[float], isIneq: List[bool]) \
        -> Optional[Tuple[np.ndarray, float, List[int], List[float]]]:
    result = solve_maximize_leq(full_matrix, list(map(lambda x: -1.0 * x, f)), isIneq)
    if result is None:
        return None
    opt, val, pos, vec = result
    return opt, -val, pos, vec
