import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from create_problem import *
from solve_flo import FLO_Solver

sns.set()  # for plot styling


def example1():
    p = problem1()
    b = FLO_Solver(p)
    print(b.solution)


def example2():
    """ example with increasing gamma """
    p = problem2()
    b = FLO_Solver(p)
    print(b.solution.log())


def example3():
    """
    compare 2 problem. one with no CL-constraint and one with a CL-constraint between the points in the middle

    0            0
      \        /
    0 - 0 -- 0 - 0
      /        \
    0            0

    """
    print('problem 3')
    p1, p2 = problem3()
    print('-> without B')
    b = FLO_Solver(p1)
    print(b.solution)
    print('-> with B')
    b = FLO_Solver(p2)
    print(b.solution)


def example4():
    p = problem4()
    b = FLO_Solver(p)
    print(b.solution.log())
    print('---------------->')
    print(b.solution)


# synthetic data
def example5():
    """
    testing algorithm on synthetic data with different CL and ML constraints
    The generated problem in the paper is saved to data/synthetic_data/problem.p
    so that we are able to reobtain the results in the paper
    """
    # p, geo_c = problem6()
    # pickle.dump(p, open("data/synthetic_data/problem.p", "wb"))
    # pickle.dump(geo_c, open("data/synthetic_data/geo.p", "wb"))
    path = "data/synthetic_data/"
    p = pickle.load(open("data/synthetic_data/problem.p", "rb"))
    geo_c = pickle.load(open("data/synthetic_data/geo.p", "rb"))

    # problem
    plot_synthetic(geo_c, name=path + 'problem')
    plot_synthetic_with_number(geo_c, name=path + 'for_finding_clients')

    # without constraints
    flo = FLO_Solver(p)
    print('flo', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, name=path + 'without-constraint')

    # with must link 1
    s = [
        [53, 4]
    ]
    S = [
        [0] * p.n
    ]
    for x in s[0]:
        S[0][x] = 1
    p = Problem(p.D, p.C, p.B, S, p.l)
    flo = FLO_Solver(p)
    print(flo.solution.exe)
    print('flo with must link 1', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, must_link=s,
                               name=path + 'with_must_link_constraint1')

    # with must link 3
    s = [
        [73, 85],

    ]
    S = [
        [0] * p.n,

    ]
    for x in s[0]:
        S[0][x] = 1

    p = Problem(p.D, p.C, p.B, S, p.l)
    flo = FLO_Solver(p)
    print('flo with must link 3', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, must_link=s,
                               name=path + 'with_must_link_constraint3')

    # with can not link1
    s = [
        [73, 85],

    ]
    S = [
        [0] * p.n,

    ]
    for x in s[0]:
        S[0][x] = 1

    b = [
        [84, 77],
    ]
    B = [
        [0] * p.n,
    ]
    for x in b[0]:
        B[0][x] = 1

    p = Problem(p.D, p.C, B, S, p.l)
    flo = FLO_Solver(p)
    print('flo with can not link1', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, can_not_link=b, must_link=s,
                               name=path + 'with_can_not_link1')


def example6():
    """ testing algorithm on synthetic data with more CL and ML constraints """
    # p, geo_c = problem6()
    # pickle.dump(p, open("data/synthetic_data/problem.p", "wb"))
    # pickle.dump(geo_c, open("data/synthetic_data/geo.p", "wb"))
    path = "data/synthetic_data/"
    p = pickle.load(open("data/synthetic_data/problem.p", "rb"))
    geo_c = pickle.load(open("data/synthetic_data/geo.p", "rb"))

    # problem
    plot_synthetic(geo_c, name=path + 'problem')
    plot_synthetic_with_number(geo_c, name=path + 'for_finding_clients')

    # without constraints
    flo = FLO_Solver(p)
    print('flo', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, name=path + 'without-constraint')

    # with must link 1
    s = [
        [53, 4]
    ]
    S = [
        [0] * p.n
    ]
    for x in s[0]:
        S[0][x] = 1
    p = Problem(p.D, p.C, p.B, S, p.l)
    flo = FLO_Solver(p)
    print(flo.solution.exe)
    print('flo with must link 1', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, must_link=s,
                               name=path + 'with_must_link_constraint1')

    # with must link 2
    s = [
        [4, 11]
    ]
    S = [
        [0] * p.n
    ]
    for x in s[0]:
        S[0][x] = 1
    p = Problem(p.D, p.C, p.B, S, p.l)
    flo = FLO_Solver(p)
    print('flo with must link 2', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, must_link=s,
                               name=path + 'with_must_link_constraint2')

    # with must link 3
    s = [
        [73, 85],

    ]
    S = [
        [0] * p.n,

    ]
    for x in s[0]:
        S[0][x] = 1

    p = Problem(p.D, p.C, p.B, S, p.l)
    flo = FLO_Solver(p)
    print('flo with must link 3', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, must_link=s,
                               name=path + 'with_must_link_constraint3')

    # with must link 4
    s = [
        [25, 94],

    ]
    S = [
        [0] * p.n,

    ]
    for x in s[0]:
        S[0][x] = 1

    p = Problem(p.D, p.C, p.B, S, p.l)
    flo = FLO_Solver(p)
    print('flo with must link 4', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, must_link=s,
                               name=path + 'with_must_link_constraint4')

    # with can not link1
    b = [
        [84, 77],
    ]
    B = [
        [0] * p.n,
    ]
    for x in b[0]:
        B[0][x] = 1

    p = Problem(p.D, p.C, B, [], p.l)
    flo = FLO_Solver(p)
    print('flo with can not link1', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, can_not_link=b,
                               name=path + 'with_can_not_link1')

    # with can not link2
    b = [
        [84, 77],

    ]
    B = [
        [0] * p.n,

    ]

    for x in b[0]:
        B[0][x] = 1
    p = Problem(p.D, p.C, B, [], p.l)
    flo = FLO_Solver(p)
    print('flo with can not link2', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, can_not_link=b,
                               name=path + 'with_can_not_link2')

    # with can not link3
    b = [
        [90, 0],
        [90, 59]

    ]
    B = [
        [0] * p.n,
        [0] * p.n,
    ]
    for x in b[0]:
        B[0][x] = 1
    for x in b[1]:
        B[1][x] = 1
    p = Problem(p.D, p.C, B, [], p.l)
    flo = FLO_Solver(p)
    print('flo with can not link1', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, can_not_link=b,
                               name=path + 'with_can_not_link3')

    # with can not link3
    b = [
        [27, 56],
    ]
    B = [
        [0] * p.n,
    ]
    for x in b[0]:
        B[0][x] = 1

    p = Problem(p.D, p.C, B, [], p.l)
    flo = FLO_Solver(p)
    print('flo with can not link4', 'dual', flo.solution.dual_value, '\t primal', flo.solution.primal_value)
    print('duality gap', (flo.solution.primal_value - flo.solution.dual_value) / flo.solution.dual_value)
    plot_synthetic_with_answer(geo_c, flo.solution.exe, flo.solution.assigns, can_not_link=b,
                               name=path + 'with_can_not_link4')


def example7():
    """ running the problem on 1000 random generated problem to find mean of duality gap """
    max_d = (0, 0, 0)  # gap, primal, dual
    gaps = []
    nn = 1000
    total = 0
    for i in range(nn):

        p, geo_c = problem6()
        import random
        random_points = random.sample(range(100), 70)
        ml = random_points[:40]
        cnl = random_points[41:]
        S = []

        for i in range(10):
            temp = [0] * p.n
            for x in ml[i * 4:i * 4 + 4]:
                temp[x] = 1
            S.append(temp)

        B = []

        for i in range(10):
            temp = [0] * p.n
            for x in cnl[i * 3:i * 3 + 2]:
                temp[x] = 1
            B.append(temp)

        p = Problem(p.D, p.C, B, S, p.l)
        try:
            flo = FLO_Solver(p)
        except Exception as e:
            print(e)
            continue
        total += 1
        primal = flo.solution.primal_value
        dual = flo.solution.dual_value
        if dual * 3 < primal:
            # we didn't witness any problem that violates 3-approximation
            pickle.dump(p, open("error.p", "wb"))
            pickle.dump(geo_c, open("geo_error.p", "wb"))
            raise Exception('it is not 3 approximate :(')
        gap = (primal - dual) / dual
        if gap > max_d[0]:
            max_d = (gap, primal, dual)
        gaps.append(gap)

    print(f"maximum duality gap:{max_d[0]} with primal={max_d[1]} and dual={max_d[2]}")
    print(sum(gaps) / total)
    print(total)


# mnist dataset
def example8():
    """ running FLOX on mnist dataset with 5 CL-constraints """
    path = 'data/mnist_dataset/'
    p, imges, labels = problem11()

    b = [
        [0] * p.n,
        [0] * p.n,
        [0] * p.n,
        [0] * p.n,
        [0] * p.n,
    ]
    b[0][116] = b[0][150] = 1
    b[1][408] = b[1][395] = 1
    b[2][230] = b[2][253] = 1
    b[3][36] = b[3][12] = 1
    b[4][114] = b[4][250] = 1

    p = Problem(p.D, p.C, b, p.S, p.l)

    flo = FLO_Solver(p)
    print(f'primal {flo.solution.primal_value} - dual {flo.solution.dual_value}')
    exe = list(flo.solution.exe)
    x = {i: [] for i in exe}
    outliers = []

    for i, e in flo.solution.assigns.items():
        if e == 'o':
            outliers.append(i)
        else:
            x[e].append(i)

    print('primal', flo.solution.primal_value, 'dual', flo.solution.dual_value)
    print(f"outliers {outliers}")
    print(f"exemplars {x.keys()}")
    print(f'number of clusters {len(exe)}')
    print(f'purity {all_purity(x, labels)}')

    yy = {i: maxFreq([labels[j][0] for j in x[i]]) for i in exe}

    print('ratio', flo.solution.primal_value / flo.solution.dual_value)
    print('number of clusters', len(flo.solution.exe))

    from sklearn.metrics.cluster import v_measure_score
    predicted = [yy[flo.solution.assigns[i]] for i in range(p.n) if flo.solution.assigns[i] != 'o']
    t = [labels[i][0] for i in range(p.n) if flo.solution.assigns[i] != 'o']
    print('v-measure', v_measure_score(t, predicted))
    from sklearn.metrics.cluster import homogeneity_score
    print('homogeneity', homogeneity_score(t, predicted))
    from sklearn.metrics.cluster import completeness_score
    print('completeness', completeness_score(t, predicted))
    print('')

    plot_mnist_data(imges[outliers], name=path + "outliers")
    plot_mnist_data(imges[exe], name=path + "exemplars")
    for i in x:
        print(i, labels[i], len(x[i]), )
        plot_mnist_data(imges[x[i]], name=path + str(i) + '-' + str(labels[i]) + '.png')


# iris dataset
def example9():
    """ iris dataset """
    p, y = problem20()
    b = [
        [0] * 150,
        [0] * 150,
    ]
    b[0][101] = b[0][50] = 1
    b[1][123] = b[1][58] = 1

    p = Problem(p.D.copy(), p.C.copy(), b, [], 5)
    flo = FLO_Solver(p)

    exe = list(flo.solution.exe)
    x = {i: [] for i in exe}
    outliers = []
    for i, e in flo.solution.assigns.items():
        if e == 'o':
            outliers.append(i)
        else:
            x[e].append(i)

    yy = {i: maxFreq([y[j] for j in x[i]]) for i in exe}
    print('purity', all_purity_iris(x, y))
    print('ratio', flo.solution.primal_value / flo.solution.dual_value)
    print('number of clusters', len(flo.solution.exe))

    from sklearn.metrics.cluster import v_measure_score
    predicted = [yy[flo.solution.assigns[i]] for i in range(p.n) if flo.solution.assigns[i] != 'o']
    t = [y[i] for i in range(p.n) if flo.solution.assigns[i] != 'o']
    print('v-measure', v_measure_score(t, predicted))
    from sklearn.metrics.cluster import homogeneity_score
    print('homogeneity', homogeneity_score(t, predicted))
    from sklearn.metrics.cluster import completeness_score
    print('completeness', completeness_score(t, predicted))


def seeds():
    """ seeds dataset """
    data = pd.read_csv('data/seed/Seed_Data.csv')
    x = data.drop(columns='target')
    y = data[['target']]
    n = len(y)

    d = [[round(euclidean(x.loc[i], x.loc[j]) * 10) for j in range(n)] for i in range(n)]

    c = [150 for i in range(n)]
    b = [
        [0] * n,
        [0] * n,
        [0] * n,
        [0] * n,
        [0] * n,
    ]
    b[0][26] = b[0][208] = 1
    b[1][27] = b[1][201] = 1
    b[2][62] = b[2][204] = 1
    b[3][63] = b[3][209] = 1
    b[4][69] = b[4][206] = 1

    s = []
    # s[0][131] = s[0][133] = 1
    # s[1][144] = s[1][182] = 1

    l = 5
    p = Problem(d, c, b, s, l)
    flo = FLO_Solver(p)
    y = [y.loc[i][0] for i in range(n)]
    info(flo, y, p, 3)


def maxFreq(arr):
    # Using moore's voting algorithm
    n = len(arr)
    res = 0
    count = 1

    for i in range(1, n):
        if arr[i] == arr[res]:
            count += 1
        else:
            count -= 1

        if count == 0:
            res = i
            count = 1

    return arr[res]


def plot_synthetic(geo_c, name=''):
    plt.scatter(geo_c[:, 0], geo_c[:, 1], s=50)
    if name:
        plt.savefig(name + '.png')
    plt.show()


def plot_synthetic_with_number(geo_c, name=''):
    plt.scatter(geo_c[:, 0], geo_c[:, 1], s=50)
    for i in range(len(geo_c)):
        plt.text(geo_c[i, 0], geo_c[i, 1], str(i), color='black')
    if name:
        plt.savefig(name + '.png')
    plt.show()


def plot_synthetic_with_answer(geo_c, exe, assign, must_link=None, can_not_link=None, name=''):
    if must_link:
        for row in must_link:
            plt.plot(geo_c[row, 0], geo_c[row, 1], c='green', alpha=0.5, linestyle='--')
    if can_not_link:
        for row in can_not_link:
            plt.plot(geo_c[row, 0], geo_c[row, 1], c='red', alpha=0.5, linestyle='--')

    x = {i: [] for i in exe}
    outliers = []
    for i, e in assign.items():
        if e == 'o':
            outliers.append(i)
        else:
            x[e].append(i)

    plt.scatter(geo_c[outliers, 0], geo_c[outliers, 1], s=50, c='red', marker="v")

    exe = list(exe)

    plt.scatter(geo_c[exe, 0], geo_c[exe, 1], c='yellow', s=200, alpha=0.5)
    # colors = iter(cm.rainbow(np.linspace(0, 1, len(exe))))
    for cluster in x.keys():
        # color = next(colors)
        plt.scatter(geo_c[x[cluster], 0], geo_c[x[cluster], 1], s=50, )

    if name:
        plt.savefig(name + '.png')
    plt.show()


def all_purity(x, labels):
    total = 0
    pur = 0
    for i in x.values():
        total += len(i)
        pur += purity(i, labels) * len(i)
    return pur / total


def purity(arr, labels):
    aa = [labels[i][0] for i in arr]  # aa = [labels[i] for i in arr]
    c = [aa.count(i) for i in range(10)]  # c = [aa.count(i) for i in range(3)]
    return max(c) / len(aa)


def all_purity_iris(x, labels):
    total = 0
    pur = 0
    for i in x.values():
        total += len(i)
        pur += purity_iris(i, labels) * len(i)
    return pur / total


def purity_iris(arr, labels):
    aa = [labels[i] for i in arr]
    c = [aa.count(i) for i in range(3)]
    return max(c) / len(aa)


def plot_mnist_data(data, r=10, c=10, name=""):
    from matplotlib import pyplot
    data = data[:100]
    for i in range(len(data)):
        pyplot.subplot(r, c, i + 1)
        img = data[i].reshape((28, 28))
        pyplot.imshow(img, cmap='gray')
        pyplot.axis('off')

    pyplot.subplots_adjust(wspace=0, hspace=0)
    if name:
        pyplot.savefig(name)
    pyplot.show()


def info(flo, y, p, clustersSize):
    exe = list(flo.solution.exe)
    x = {i: [] for i in exe}
    outliers = []
    for i, e in flo.solution.assigns.items():
        if e == 'o':
            outliers.append(i)
        else:
            x[e].append(i)

    yy = {i: maxFreq([y[j] for j in x[i]]) for i in exe}
    print('purity', all_purity_seed(x, y, clustersSize))
    print('ratio', flo.solution.primal_value / flo.solution.dual_value)
    print('number of clusters', len(flo.solution.exe))

    from sklearn.metrics.cluster import v_measure_score
    predicted = [yy[flo.solution.assigns[i]] for i in range(p.n) if flo.solution.assigns[i] != 'o']
    t = [y[i] for i in range(p.n) if flo.solution.assigns[i] != 'o']
    print('v-measure', v_measure_score(t, predicted))
    from sklearn.metrics.cluster import homogeneity_score
    print('homogeneity', homogeneity_score(t, predicted))
    from sklearn.metrics.cluster import completeness_score
    print('completeness', completeness_score(t, predicted))
    print('')


def all_purity_seed(x, labels, clustersSize):
    total = 0
    pur = 0
    for i in x.values():
        total += len(i)
        pur += purity_seed(i, labels, clustersSize) * len(i)
    return pur / total


def purity_seed(arr, labels, clustersSize):
    aa = [labels[i] for i in arr]  # aa = [labels[i][0] for i in arr]
    c = [aa.count(i) for i in range(clustersSize)]  # c = [aa.count(i) for i in range(10)]
    return max(c) / len(aa)
