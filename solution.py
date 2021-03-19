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
    solution = find_acceptable_solution(full_matrix)
    if solution is None:
        return None
    positions, vec = solution
    point, res = solve_maximize(full_matrix, f, positions)
    return point, res, positions, vec


def full_solve_minimize(full_matrix: List[List[float]], f: List[float]) -> Optional[Tuple[np.ndarray, float, List[int], List[float]]]:
    solution = full_solve_maximize(full_matrix, (np.array(f) * -1).tolist())
    if solution is None:
        return None
    point, val, pos, vec = solution
    return point, -val, pos, vec
