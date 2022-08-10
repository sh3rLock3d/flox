import numpy as np


class FL_Solver:
    def __init__(self, p):
        self.p = p
        self.find_dual(self.p)

        self.primal_solution()

        self.solution = Solution(self)

    def find_dual(self, p):
        alpha = np.array([0] * p.n)
        beta = np.array([[0] * p.n] * p.n)

        inactive_clients = set(range(p.n))
        facilities_paid_for = []

        while inactive_clients:
            new_active_clients = set()
            for j in inactive_clients:
                tight_edges = np.where(alpha[j] >= p.D[:, j])[0]

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
        self.alpha, self.beta, self.facilities_paid_for = alpha, beta, facilities_paid_for

    def primal_solution(self):
        p = self.p
        alpha = self.alpha
        beta = self.beta

        facilities_paid_for = self.facilities_paid_for

        client_edges = {i: [] for i in range(p.n)}
        facility_edges = {i: [] for i in facilities_paid_for}
        for i in range(p.n):
            if i not in facilities_paid_for:
                continue
            for j in range(p.n):
                if alpha[j] - beta[i, j] >= p.D[i, j]:
                    # they are connected
                    client_edges[j].append(i)
                    facility_edges[i].append(j)

        open_facilities = set()
        connected_clients = dict()

        while facilities_paid_for:
            i = facilities_paid_for.pop(0)
            # distance one
            distance1 = set()
            for j in facility_edges[i]:
                if j not in connected_clients.keys():
                    distance1.add(j)

            open_facilities.add(i)

            distance2 = set()
            distance3 = set()
            for j in distance1:
                facility_at_d2 = set(client_edges[j]) - {i}
                for i_d2 in facility_at_d2:
                    distance2.add(i_d2)
                    connected = set(facility_edges[i_d2]) - distance1 - connected_clients.keys()
                    distance3.update(connected)

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

        self.open_facilities = open_facilities
        self.connected_clients = connected_clients


class Solution:
    def __init__(self, solve_flo):
        self.p = solve_flo.p

        self.alpha = solve_flo.alpha
        self.beta = solve_flo.beta

        self.facilities_paid_for = solve_flo.facilities_paid_for
        self.connected_clients = solve_flo.connected_clients
        self.exemplar = solve_flo.open_facilities

        self.primal_value = sum([self.p.D[key, value] for key, value in self.connected_clients.items()]) + sum(
            [self.p.C[i] for i in self.exemplar])
        self.dual_value = sum(solve_flo.alpha)
        self.assigns = solve_flo.connected_clients.copy()

    def log(self):
        res = ''
        res += '----------problem' + '\n'
        res += str(self.p) + '\n'
        res += '---------------dual' + '\n'
        res += 'alpha' + '\n'
        res += str(self.alpha) + '\n'
        res += 'beta' + '\n'
        res += str(self.beta) + '\n'
        res += 'facilities paid for' + '\n'
        res += str(self.facilities_paid_for) + '\n'
        res += '---------------primal' + '\n'
        res += 'connected_clients' + '\n'
        res += str(self.connected_clients) + '\n'
        res += 'exemplars' + '\n'
        res += str(self.exemplar) + '\n'

        res += 'assign' + '\n'
        res += str(self.assigns) + '\n'

        res += '-------------' + '\n'
        res += 'dual value :' + str(np.sum(self.alpha)) + '\n'
        res += 'primal value :' + str(
            sum([self.p.D[key, value] for key, value in self.connected_clients.items()]) + sum(
                [self.p.C[i] for i in self.exemplar]))
        return res

    def __str__(self):
        res = ''
        res += 'assign' + '\n'
        res += str(self.assigns) + '\n'

        res += '-------------' + '\n'
        res += 'dual value :' + str(np.sum(self.alpha)) + '\n'
        res += 'primal value :' + str(
            sum([self.p.D[key, value] for key, value in self.connected_clients.items()]) + sum(
                [self.p.C[i] for i in self.exemplar]))
        return res

