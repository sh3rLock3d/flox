import numpy as np
from solve_fl import FL_Solver


class Problem:
    def __init__(self, D, C):
        self.D = np.array(D)  # distance between j and i
        self.C = np.array(C)  # cost of a point to be the exemplar of a cluster
        # S and B must be sorted (it means that for row i < j first nonzero index should be less than the one in row j)
        self.n = len(D)  # number of points

    def __str__(self):
        x = ''
        x += 'D' + '\n' + str(self.D) + '\n'
        x += 'C' + '\n' + str(self.C) + '\n'
        return x


def problem1():
    x = [-4, -3, -2, 0, 2, 3, 4]
    n = len(x)
    D = [[abs(x[i] - x[j]) for i in range(n)] for j in range(n)]
    C = [3 for _ in range(n)]

    p = Problem(D, C)
    return p


p = problem1()
b = FL_Solver(p)
print(b.solution)
# print(b.solution.log())
