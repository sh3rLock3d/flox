import numpy as np

from create_problem import Problem


class FLO_Solver:
    def __init__(self, p):
        self.p = p
        self.outliers = set()
        self.first_step()

        self.index_s = None
        self.index = None
        self.newP = None
        self.second_step()

        self.alpha, self.beta, self.gamma, self.facilities_paid_for = self.find_dual(self.newP)

        self.forth_step()

        self.solution = Solution(self)
        self.solution.validate()

    def first_step(self):
        p = self.p
        alpha, _, _, _ = self.find_dual(p)

        cant_be_outlier = set()
        for r in p.S:
            a = np.nonzero(r)[0]
            cant_be_outlier.update(a)
        largest_alpha = sorted(range(p.n), key=lambda k: alpha[k], reverse=True)
        largest_alpha = [x for x in largest_alpha if x not in cant_be_outlier]
        self.outliers = set(largest_alpha[:p.l])

    def second_step(self):
        p = self.p
        newD = p.D.copy()
        newC = p.C.copy()
        newB = p.B.copy()
        index = list(range(p.n))
        index_s = dict()
        for r in p.S:
            a = np.nonzero(r)[0]
            indexOfClient = [index.index(i) for i in a]
            i = indexOfClient[0]
            for j in range(len(newC)):
                if i == j:
                    newD[i, i] = sum(p.D[k, j] for k in indexOfClient[0:])
                else:
                    newD[i, j] = newD[j, i] = sum(p.D[k, j] for k in indexOfClient)
            for x in a[1:]:
                index_s[x] = a[0]
            newD = np.delete(newD, indexOfClient[1:], axis=1)
            newD = np.delete(newD, indexOfClient[1:], axis=0)
            newC = np.delete(newC, indexOfClient[1:])
            if len(newB):
                newB = np.delete(newB, indexOfClient[1:], axis=1)
            index = [i for i in index if i not in a[1:]]
        # todo if there is a rule that can be deleted, delete it.
        # r[~np.all(r == 0, axis=1)]
        self.newP = Problem(newD, newC, newB, None, 0)
        self.index = index
        self.index_s = index_s

        new_outliers = set()
        for i in self.outliers:
            new_outliers.add(index.index(i))
        self.outliers = new_outliers

    def find_dual(self, p):  # third step
        alpha = np.array([0] * p.n)
        beta = np.array([[0] * p.n] * p.n)
        gamma = np.array([[0] * p.n] * p.r)

        inactive_clients = set(range(p.n)) - set(self.outliers)
        facilities_paid_for = []

        while inactive_clients:
            new_active_clients = set()
            for j in inactive_clients:
                tight_edges = np.where(alpha[j] >= p.D[:, j])[0]  # saturated edges
                tight_edges = [i for i in tight_edges if i not in self.outliers]
                if any(i in facilities_paid_for for i in tight_edges):
                    # can't increase
                    new_active_clients.add(j)
                else:
                    # can increase
                    alpha[j] += 1
                    for i in tight_edges:
                        beta[i, j] += 1
                        if i not in facilities_paid_for and sum(beta[i, :]) == p.C[i]:
                            facilities_paid_for.append(i)
            inactive_clients = inactive_clients - new_active_clients
        '''
        print('alpha')
        print(alpha)
        print('beta')
        print(beta)
        print('gamma')
        print(gamma)
        print('pa')
        print(facilities_paid_for)
        '''

        # gamma part
        for r in range(p.r):
            row = p.B[r]
            x = np.nonzero(row)[0]
            while True:
                gamma_changed = False
                components = []  # every element is a tuple: set of clients, set of facilities
                for j in x:
                    if j in self.outliers:
                        continue
                    connected_to_j = [ii for ii in range(p.n) if
                                      alpha[j] - beta[ii, j] - sum(
                                          [p.B[rr, j] * gamma[rr, ii] for rr in range(p.r)]) >= p.D[
                                          ii, j]]
                    connected_to_j = [ii for ii in connected_to_j if ii in facilities_paid_for]

                    for component in components:
                        if any(ii in component[1] for ii in connected_to_j):
                            component[0].add(j)
                            component[1].update(set(connected_to_j))
                            break
                    else:
                        components.append((
                            {j},
                            set(connected_to_j)
                        ))

                for component in components:
                    clients = component[0]
                    facilities = component[1]
                    if len(clients) > len(facilities):
                        for j in clients:
                            alpha[j] += 1
                        for i in facilities:
                            gamma[r][i] += 1
                        gamma_changed = True

                        for j in clients:
                            for i in range(p.n):
                                if i in self.outliers:
                                    continue
                                if alpha[j] >= p.D[i, j]:
                                    beta[i, j] = alpha[j] - p.D[i, j] - sum(
                                        [p.B[rr, j] * gamma[rr, i] for rr in range(p.r)])
                                    if beta[i, j] < 0:
                                        print('here')
                                    if i not in facilities_paid_for and sum(beta[i, :]) == p.C[i]:
                                        facilities_paid_for.append(i)
                        break
                if not gamma_changed:
                    break

        '''
        print('alpha')
        print(alpha)
        print('beta')
        print(beta)
        print('gamma')
        print(gamma)
        print('pa')
        print(facilities_paid_for)
        print('------')
        self.find_connected_to_client(p,alpha,beta,  gamma)
        '''

        return alpha, beta, gamma, facilities_paid_for

    def find_connected_to_facility(self, p, alpha, beta, gamma):
        print('facilities')
        for i in range(p.n):
            if i in self.outliers:
                continue
            connected = [j for j in range(p.n) if
                         alpha[j] - beta[i, j] - sum([p.B[r, j] * gamma[r, i] for r in range(p.r)]) >= p.D[i, j]]
            connected = [j for j in connected if j not in self.outliers]
            print(f'{i}-->{connected}')

    def find_connected_to_client(self, p, alpha, beta, gamma):
        print('clients')
        for j in range(p.n):
            if j in self.outliers:
                continue
            connected = [i for i in range(p.n) if
                         alpha[j] - beta[i, j] - sum([p.B[r, j] * gamma[r, i] for r in range(p.r)]) >= p.D[i, j]]
            connected = [i for i in connected if i not in self.outliers]
            print(f'{j}-->{connected}')

    def forth_step(self):
        p = self.newP
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        facilities_paid_for = self.facilities_paid_for
        '''
        print('alpha')
        print(alpha)
        print('beta')
        print(beta)
        print('gamma')
        print(gamma)
        print('pa')
        print(facilities_paid_for)
        print('------')
        '''
        client_edges = {i: [] for i in range(p.n) if i not in self.outliers}
        facility_edges = {i: [] for i in facilities_paid_for}
        for i in range(p.n):
            if i not in facilities_paid_for:
                continue
            for j in range(p.n):
                if j in self.outliers:
                    continue
                if alpha[j] - beta[i, j] - sum([p.B[r, j] * gamma[r, i] for r in range(p.r)]) >= p.D[i, j]:
                    # they are connected
                    client_edges[j].append(i)
                    facility_edges[i].append(j)

        for r in range(p.r):
            row = p.B[r]
            x = np.nonzero(row)[0]  # todo implementation is for pairwise constraints
            j = x[0]
            jp = x[1]
            for i in facilities_paid_for:
                if j in facility_edges[i] and jp in facility_edges[i]:
                    deleted_j = j
                    if len(client_edges[j]) == len(client_edges[jp]):
                        if p.D[i, j] < p.D[i, jp]:
                            deleted_j = jp
                    elif len(client_edges[j]) < len(client_edges[jp]):
                        deleted_j = jp

                    facility_edges[i].remove(deleted_j)
                    client_edges[deleted_j].remove(i)

        '''
        print(client_edges)
        print(facility_edges)
        '''
        open_facilities = set()
        connected_clients = dict()

        while facilities_paid_for:
            i = facilities_paid_for.pop(0)
            should_not_be_in_cluster = set()
            # distance one
            distance1 = set()
            for j in facility_edges[i]:
                if j not in connected_clients.keys():
                    distance1.add(j)
                    should_not_be_in_cluster.update(self.should_not_be_in_the_same_cluster_with(j))
            if len(distance1) == 0:
                continue  # I don't think this happens, but to be sure let's check it
            open_facilities.add(i)

            distance2 = set()
            distance3 = set()
            for j in distance1:
                facility_at_d2 = set(client_edges[j]) - {i}
                for i_d2 in facility_at_d2:
                    connected = set(facility_edges[i_d2]) - distance1 - connected_clients.keys()
                    if connected.intersection(should_not_be_in_cluster):
                        # they have intersection
                        distance3 = distance3 - connected
                        continue
                    else:
                        distance2.add(i_d2)
                        distance3.update(connected)
                        for jj in connected:
                            should_not_be_in_cluster.update(self.should_not_be_in_the_same_cluster_with(jj))
            # delete facilities at distance 2
            facilities_paid_for = [ii for ii in facilities_paid_for if ii not in distance2]
            for x in distance2:
                facility_edges.pop(x, None)
            for x, y in client_edges.items():
                for z in distance2:
                    if z in y:
                        client_edges[x].remove(z)
            # assign
            for j in distance1:
                connected_clients[j] = i
            for j in distance3:
                connected_clients[j] = i

        '''
        print(open_facilities)
        print(connected_clients)
        '''

        self.open_facilities = open_facilities
        self.connected_clients = connected_clients

    def should_not_be_in_the_same_cluster_with(self, j):
        b = self.newP.B
        if len(b) == 0:
            # doesn't have any rules
            return set()
        x = np.nonzero(b[:, j])[0]
        if len(x) == 0:
            return set()
        r = x[0]
        x = set(np.nonzero(b[r, :])[0])
        x = x - {j}
        return x



class Solution:
    def __init__(self, solve_flo):
        self.p = solve_flo.p
        self.newP = solve_flo.newP

        self.alpha = solve_flo.alpha
        self.beta = solve_flo.beta

        self.gamma = solve_flo.gamma

        self.facilities_paid_for = solve_flo.facilities_paid_for
        self.connected_clients = solve_flo.connected_clients
        self.exemplar = solve_flo.open_facilities
        self.outliers = solve_flo.outliers

        self.dual_value = np.sum(self.alpha) - np.sum(self.gamma)
        self.primal_value = sum([self.newP.D[key, value] for key, value in self.connected_clients.items()]) + sum(
            [self.newP.C[i] for i in self.exemplar])

        self.assigns = dict()
        for i in range(self.p.n):
            if i in solve_flo.index_s.keys():
                x = solve_flo.index_s[i]
                x = solve_flo.index.index(x)
                x = self.connected_clients[x]
                x = solve_flo.index[x]
                self.assigns[i] = x
            elif solve_flo.index.index(i) in self.outliers:
                self.assigns[i] = 'o'
            else:
                x = solve_flo.index.index(i)
                x = self.connected_clients[x]
                x = solve_flo.index[x]
                self.assigns[i] = x
        self.exe = set()
        for i in self.exemplar:
            self.exe.add(
                solve_flo.index[i]
            )

    def log(self):
        res = ''
        res += '----------problem' + '\n'
        res += 'p' + '\n'
        res += str(self.p) + '\n'
        res += 'new p' + '\n'
        res += str(self.newP) + '\n'
        res += '---------------dual' + '\n'
        res += 'alpha' + '\n'
        res += str(self.alpha) + '\n'
        res += 'beta' + '\n'
        res += str(self.beta) + '\n'
        res += 'gamma' + '\n'
        res += str(self.gamma) + '\n'
        res += 'facilities paid for' + '\n'
        res += str(self.facilities_paid_for) + '\n'
        res += '---------------primal' + '\n'
        res += 'connected_clients' + '\n'
        res += str(self.connected_clients) + '\n'
        res += 'exemplars' + '\n'
        res += str(self.exemplar) + '\n'
        res += 'outliers' + '\n'
        res += str(self.outliers) + '\n'

        res += 'exe' + '\n'
        res += str(self.exe) + '\n'

        res += 'assign' + '\n'
        res += str(self.assigns) + '\n'

        res += '-------------' + '\n'
        res += 'dual value :' + str(np.sum(self.alpha) - np.sum(self.gamma)) + '\n'
        res += 'primal value :' + str(
            sum([self.newP.D[key, value] for key, value in self.connected_clients.items()]) + sum(
                [self.newP.C[i] for i in self.exemplar]))
        return res

    def __str__(self):
        res = ''
        res += 'outliers' + '\n'
        res += str(self.outliers) + '\n'

        res += 'exe' + '\n'
        res += str(self.exe) + '\n'

        res += 'assign' + '\n'
        res += str(self.assigns) + '\n'

        res += '-------------' + '\n'
        res += 'dual value :' + str(np.sum(self.alpha) - np.sum(self.gamma)) + '\n'
        res += 'primal value :' + str(
            sum([self.newP.D[key, value] for key, value in self.connected_clients.items()]) + sum(
                [self.newP.C[i] for i in self.exemplar]))
        return res

    def validate(self):
        if np.any(self.beta<0):
            raise Exception("beta has negative value")
        if self.dual_value < 0:
            raise Exception("beta is negative")
        if self.dual_value > self.primal_value:
            raise Exception("dual is grater than primal")

        if self.assigns.keys() != set(range(self.p.n)):
            raise Exception("some clients are not assigned")

        for r in range(self.p.r):
            row = self.p.B[r]
            x = np.nonzero(row)[0]
            if self.assigns[x[0]] == self.assigns[x[1]] and self.assigns[x[0]] != 'o':
                raise Exception(f'{x[0]} and {x[1]} are both connected to {self.assigns[x[0]]}')

        for r in range(len(self.p.S)):
            row = self.p.S[r]
            x = np.nonzero(row)[0]
            if self.assigns[x[0]] != self.assigns[x[0]]:
                raise Exception(f'{x[0]} and {x[1]} are in differant clusters')

