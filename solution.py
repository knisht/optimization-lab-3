from typing import Optional, List, Tuple

import numpy as np

from matrix_preparation import prepare_matrix
from simplex import simplex_method, find_acceptable_solution


def solve_maximize(full_matrix: List[List[float]], f: List[float], positions: List[int]) -> \
        Optional[Tuple[np.ndarray, float]]:
    prepared_matrix = prepare_matrix(full_matrix, positions)
    result = simplex_method(prepared_matrix, np.array(f), positions)
    if result is None:
        return None
    opt_sol, pos, val = result
    print("Base: ", pos)
    return opt_sol, val


def solve_minimize(full_matrix: List[List[float]], f: List[float], positions: List[int]) \
        -> Optional[Tuple[np.ndarray, float]]:
    result = solve_maximize(full_matrix, (np.array(f) * -1).tolist(), positions)
    if result is None:
        return None
    opt_sol, val = result
    return opt_sol, -val


def full_solve_maximize(full_matrix: List[List[float]], f: List[float]) -> Optional[Tuple[np.ndarray, float, List[int], List[float]]]:
    positions, vec = find_acceptable_solution(full_matrix)
    if positions is None:
        return None
    point, res = solve_maximize(full_matrix, f, positions)
    return point, res, positions, vec


def full_solve_minimize(full_matrix: List[List[float]], f: List[float]) -> Optional[Tuple[np.ndarray, float, List[int], List[float]]]:
    point, val, pos, vec = full_solve_maximize(full_matrix, (np.array(f) * -1).tolist())
    return point, -val, pos, vec
