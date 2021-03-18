from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from solution import full_solve

if __name__ == '__main__':
    input = [[1, 2, 3], [4, 5, 6]]
    x0 = [1, 0, 0]
    b = [1, 2, 3]
    f = [4, 5, 6]

    full_matrix = [[2.0, 1.0, -1.0, 10.0],
                   [4.0, 1.0, 1.0, 1.0]]
    print(full_solve(full_matrix, f))

    print("hello")
