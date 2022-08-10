import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial.distance import euclidean


class Problem:
    def __init__(self, D, C, B, S, l):
        self.D = np.array(D)  # distance between j and i
        self.C = np.array(C)  # cost of a point to be the exemplar of a cluster
        # S and B must be sorted (it means that for row i < j first nonzero index should be less than the one in row j)
        self.B = np.array(B)  # rules
        self.S = np.array(S)  # points must be at the same cluster
        self.n = len(D)  # number of points
        self.r = len(B)
        self.l = l  # number of outliers

    def __str__(self):
        x = ''
        x += 'D' + '\n' + str(self.D) + '\n'
        x += 'C' + '\n' + str(self.C) + '\n'
        x += 'B' + '\n' + str(self.B) + '\n'
        x += 'S' + '\n' + str(self.S) + '\n'
        return x


def problem1():
    x = [-4, -3, -2, 0, 2, 3, 4]
    n = len(x)
    D = [[abs(x[i] - x[j]) for i in range(n)] for j in range(n)]
    C = [3 for _ in range(n)]
    B = [
        [0, 0, 0, 0, 0, 1, 1]
    ]
    S = [
        [1, 1, 0, 0, 0, 0, 0],
    ]
    l = 1
    p = Problem(D, C, B, S, l)
    return p


def problem2():
    x = [1, 2, 3, 4, 7, 8, 9]
    n = len(x)
    D = [[abs(x[i] - x[j]) for i in range(n)] for j in range(n)]
    C = [3 for _ in range(n)]
    C[4] = 10
    B = [
        [0, 0, 0, 0, 0, 1, 1],
    ]

    S = [
    ]
    l = 0
    p = Problem(D, C, B, S, l)
    return p


def problem3():
    n = 8
    D = [[0, 3, 6, 5, 6, 10, 11, 12],
         [3, 0, 3, 4, 6, 11, 10, 11],
         [6, 3, 0, 5, 6, 12, 11, 10],
         [5, 4, 5, 0, 2, 7, 6, 7],
         [7, 6, 7, 2, 0, 5, 4, 5],
         [10, 11, 12, 7, 5, 0, 3, 6],
         [11, 10, 11, 6, 4, 3, 0, 3],
         [12, 11, 10, 7, 5, 6, 3, 0]]

    C = [6 for _ in range(n)]
    B = [
    ]

    S = [
    ]
    l = 0
    p1 = Problem(D, C, B, S, l)
    B = [
        [0, 0, 0, 1, 1, 0, 0, 0],
    ]
    p2 = Problem(D, C, B, S, l)
    return p1, p2


def problem4():
    n = 8
    D = [
        [0, 14, 14, 40, 31, 31, 10, 30],
        [14, 0, 20, 31, 28, 20, 10, 22],
        [14, 20, 0, 31, 20, 28, 10, 22],
        [40, 31, 31, 0, 14, 14, 30, 10],
        [31, 28, 20, 14, 0, 20, 22, 10],
        [31, 20, 28, 14, 20, 0, 22, 10],
        [10, 10, 10, 30, 22, 22, 0, 20],
        [30, 22, 22, 10, 10, 10, 20, 0]
    ]

    C = [40 for _ in range(n)]
    B = [
        [0, 1, 0, 0, 0, 0, 1, 0],
    ]

    S = [
    ]
    l = 0
    p1 = Problem(D, C, B, S, l)
    return p1


# synthetic data
def problem6(centers=10, samples=100):
    cluster_std = 6
    geo_c, label, geo_f = make_blobs(n_samples=samples, centers=centers, cluster_std=cluster_std,
                                     random_state=None, return_centers=True, center_box=(0, 100), )
    d = []
    for i in range(samples):
        row = []
        for j in range(samples):
            row.append(euclidean(geo_c[i], geo_c[j]))
        d.append(row)
    d = np.array(d)
    d = np.rint(d)
    d = np.array(d, dtype=int)

    c = [15 for _ in range(samples)]
    p = Problem(d, c, [], [], 5)

    return p, geo_c


# mnist dataset
def problem11():
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import euclidean
    data_path = "data/mnist_dataset/"
    # train_data = np.loadtxt(data_path + "mnist_train.csv",delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
    fac = 0.99 / 255
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
    test_labels = np.asfarray(test_data[:, :1])

    # test_imgs = test_imgs[:1000]
    # test_labels = test_labels[:1000]
    test_imgs = test_imgs[:500]
    test_labels = test_labels[:500]
    pca = PCA(n_components=25)
    clients = pca.fit_transform(test_imgs)
    n = len(clients)

    c = [900 for _ in range(n)]
    d = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(euclidean(clients[i], clients[j]) * 100)
        d.append(row)
    d = np.array(d)
    d = np.rint(d)
    d = np.array(d, dtype=int)

    p = Problem(d, c, [], [], 50)
    return p, test_imgs, test_labels


def problem20():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import datasets
    from sklearn.decomposition import PCA
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n = len(X)
    d = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(euclidean(X[i], X[j]) * 10)
        d.append(row)
    c = [50 for _ in range(n)]
    d = np.array(d)
    d = np.rint(d)
    d = np.array(d, dtype=int)
    p = Problem(d, c, [], [], 0)
    return p, y
