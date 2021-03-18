from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from matrix_preparation import prepare_matrix
from solution import full_solve_maximize, solve_minimize, full_solve_minimize
from utils import format_list

if __name__ == '__main__':
    input1 = [[4.0, 3.0, 1.0, -1.0, 1.0],
              [4.0, 5.0, 1.0, 1.0, -1.0]]
    f1 = [-6.0, -1.0, -4.0, 5.0]
    pos1 = [1, 4]

    input2 = [[-4., 1., -3., -1., -2.],
              [0., 1., -1., 1., 0.]]
    f2 = [-1., -2., -3., 1.]
    pos2 = [2, 3]

    input3 = [[5., 1., 1., 0., 2., 1.],
              [9., 1., 1., 1., 3., 2.],
              [6., 0., 1., 1., 2., 1.]]
    f3 = [-1., -2., -1., 3., -1.]
    pos3 = [3, 4, 5]

    input4 = [[4., 1., 1., 2., 0., 0.],
              [-6., 0., -2., -2., 1., -1.],
              [12., 1., -1., 6., 1., 1.]]
    f4 = [-1., -1., -1., 1., -1.]
    pos4 = [1, 2, 3]

    input5 = [[0., 1., 1., -1., -10.],
              [11., 1., 14., 10., -10.]]
    f5 = [-1., 4., -3., 10.]

    input6 = [[3., 1., 3., 3., 1., 1., 0.],
              [4., 2., 0., 3., -1., 0., 1.]]
    f6 = [-1., 5., 1., -1., 0., 0.]
    pos6 = [5, 6]

    input7 = [[10., 3., 1., 1., 1., -2.],
              [20., 6., 1., 2., 3., -4.],
              [30., 10., 1., 3., 6., -7.]]
    f7 = [-1., -1., 1., -1., 2.]

    full_matrices = [input1, input2, input3, input4, input5, input6, input7]
    fs = [f1, f2, f3, f4, f5, f6, f7]

    solve_matrices = [input1, input2, input3, input4, input6]
    solve_fs = [f1, f2, f3, f4, f6]
    solve_pos = [pos1, pos2, pos3, pos4, pos6]
    solve_nums = []
    full_solve_matrices = [input5, input7]
    full_solve_fs = [f5, f7]
    full_solve_nums = [5, 7]

    # print(solve_minimize(input3, f3, pos3))
    for matrix, f, pos, num in zip(solve_matrices, solve_fs, solve_pos, solve_nums):
        point, val = solve_minimize(matrix, f, pos)
        print("Answer for num: ", num)
        if (num == 6):
            print("Resulting point: ", format_list(point.tolist()[:-2]))
        else:
            print("Resulting point: ", format_list(point.tolist()))
        print("Function value: ", val)
        print("====================")

    for matrix, f, num in zip(full_solve_matrices, full_solve_fs, full_solve_nums):
        point, val, positions, vec = full_solve_minimize(deepcopy(matrix), f)
        print("Answer for num: ", num)
        print("Resulting point: ", format_list(point.tolist()))
        print("Function value: ", val)
        print("Starting point: ", format_list(vec))
        print("Base: ", positions)
        print("====================")
        if (num == 7):
            print(prepare_matrix(matrix, [2, 3, 5]))
