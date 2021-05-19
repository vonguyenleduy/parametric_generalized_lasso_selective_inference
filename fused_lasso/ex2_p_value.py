import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import gen_data
import qp_solver
import util
import parametric_si


def compute_c_d(X, a, b, p, lamda):
    dim_beta = p
    dim_z = p - 1
    no_vars = p + 2 * dim_z

    e_1 = lamda * np.hstack((np.zeros(dim_beta), np.ones(2 * dim_z)))
    e_2 = np.hstack((np.dot(X.T, a).flatten(), np.zeros(2 * dim_z)))

    c = e_1 - e_2
    d = np.hstack((-np.dot(X.T, b).flatten(), np.zeros(2 * dim_z)))

    c = c.reshape((no_vars, 1))
    d = d.reshape((no_vars, 1))

    return c, d


def construct_P_q_G_h_A_b(X, y, D, n, p, lamda):
    dim_beta = p
    dim_z = p - 1
    no_vars = p + 2 * dim_z

    # construct P
    e_1 = np.hstack((X, np.zeros((n, 2 * dim_z))))
    P = np.dot(e_1.T, e_1)

    # construct q
    e_1 = lamda * np.hstack((np.zeros(dim_beta), np.ones(2 * dim_z)))
    e_2 = np.hstack((np.dot(X.T, y).flatten(), np.zeros(2 * dim_z)))
    q = e_1 - e_2
    q = q.reshape((no_vars, 1))

    # construct G
    G = np.zeros((no_vars, no_vars))
    G[dim_beta:, dim_beta:] = np.zeros((2 * dim_z, 2 * dim_z)) - np.identity(2 * dim_z)

    # construct h
    h = np.zeros((no_vars, 1))

    # construct A
    e_1 = np.hstack((np.identity(dim_z), np.zeros((dim_z, dim_z)) - np.identity(dim_z)))
    A = np.hstack((-D, e_1))

    # construct b
    b = np.zeros((D.shape[0], 1))

    return P, q, G, h, A, b


def run():
    n = 60
    p = n
    lamda = 15
    z_threshold = 20

    X = np.identity(n)

    D = (np.diag([-1] * n, k=0) + np.diag([1] * (n - 1), k=1))[:-1]

    true_y = np.zeros(n)
    # True CPs are at positions 20 and 40
    true_y[20:40] += 3
    y = true_y + np.random.normal(0, 1, n)

    y = y.reshape((n, 1))
    true_y = true_y.reshape((n, 1))

    P, q, G, h, A, b = construct_P_q_G_h_A_b(X, y, D, n, p, lamda)

    dim_x = p + 2 * (p - 1)
    x, prob = qp_solver.run(P, q, G, h, A, b, dim_x)

    x = x.value

    beta = x[0:p]
    list_cp = util.find_list_cp(beta, p)

    if len(list_cp) == 2:
        return None

    for cp_selected_idx in range(1, len(list_cp) - 1):
        cp_selected = list_cp[cp_selected_idx]

        pre_cp = list_cp[cp_selected_idx - 1]
        next_cp = list_cp[cp_selected_idx + 1]
        n_s_1 = cp_selected - pre_cp
        n_s_2 = next_cp - cp_selected

        etaj = np.zeros(n)
        etaj[pre_cp + 1 : cp_selected + 1] = np.ones(n_s_1) / n_s_1
        etaj[cp_selected + 1 : next_cp + 1] = - np.ones(n_s_2) / n_s_2
        etaj = etaj.reshape((n, 1))

        etajTy = np.dot(etaj.T, y)[0][0]

        a_line, b_line = util.compute_a_b(y, etaj, n)

        c, d = compute_c_d(X, a_line, b_line, p, lamda)

        path_zk, path_list_cp = parametric_si.run_parametric_si(P, c, d, G, h, A, b, p, dim_x, z_threshold)

        cov = np.identity(n)
        p_value = util.p_value(list_cp, path_zk, path_list_cp, etaj, etajTy, cov)

        print('Detected changepoint:', cp_selected + 1, ' p_value:', p_value)
        print("==========")


if __name__ == '__main__':
    run()