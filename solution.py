from typing import Optional, List, Tuple

import numpy as np

from matrix_preparation import prepare_matrix
from simplex import simplex_method, find_acceptable_solution


def solve(full_matrix: np.ndarray, f: np.ndarray, positions: List[int]) -> Optional[Tuple[np.ndarray, float]]:
    prepared_matrix = prepare_matrix(full_matrix, positions)
    tuple = simplex_method(prepared_matrix, f, positions)
    if tuple is None:
        return None
    opt_sol, _, val = tuple
    print("Optimal solution: ", opt_sol)
    print("Value of function: ", val)
    return opt_sol, val


def full_solve(full_matrix: List[List[float]], f: List[float]) -> Optional[Tuple[np.ndarray, float]]:
    full_matrix_np = np.array(full_matrix)
    positions = find_acceptable_solution(full_matrix_np)
    if positions is None:
        return None
    return solve(full_matrix_np, np.array(f), positions)
