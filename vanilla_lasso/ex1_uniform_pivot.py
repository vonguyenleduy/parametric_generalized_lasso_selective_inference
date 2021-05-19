import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import gen_data
import qp_solver
import util
import parametric_si


def construct_P_q_G_h_A_b(X, y, p, lamda):
    no_vars = 2 * p

    # construct P
    P = np.zeros((no_vars, no_vars))
    XTX = np.dot(X.T, X)
    P[0:p, 0:p] = XTX
    P[0:p, p:2 * p] = -XTX
    P[p:2 * p, 0:p] = -XTX
    P[p:2 * p, p:2 * p] = XTX

    # construct q
    e_1 = lamda * np.ones((no_vars, 1))
    XTy = np.dot(X.T, y)
    e_2 = np.zeros((no_vars, 1))
    e_2[0:p] = XTy
    e_2[p:2 * p] = -XTy
    q = e_1 - e_2

    # construct G
    G = np.zeros((2 * p, no_vars)) - np.identity(no_vars)

    # construct h
    h = np.zeros((2 * p, 1))

    return P, q, G, h, None, None


def run():
    n = 100
    p = 5
    lamda = 10
    beta_vec = np.zeros(p)
    signal = 2
    no_active = 2
    for i in range(no_active):
        beta_vec[i] = signal

    z_threshold = 20

    X, y, true_y = gen_data.generate_test(n, p, beta_vec)
    y = y.reshape((n, 1))
    true_y = true_y.reshape((n, 1))

    P, q, G, h, _, _ = construct_P_q_G_h_A_b(X, y, p, lamda)

    x, prob = qp_solver.solve_lasso(P, q, G, h, 2 * p)

    x = x.value
    x = x.reshape((len(x), 1))

    B_plus = x[0:p]
    B_minus = x[p:2*p]

    beta_hat = B_plus - B_minus
    bh = util.check_zero(beta_hat.flatten())

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, p)

    rand_value = np.random.randint(len(A))
    j_selected = A[rand_value]

    etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

    a, b = util.compute_a_b(y, etaj, n)

    c, d = util.compute_c_d(X, a, b, lamda, p)

    list_zk, list_bhz, list_active_set = parametric_si.run_parametric_si(X, P, G, h, p, c, d, z_threshold)

    tn_mu = np.dot(etaj.T, true_y)[0][0]
    cov = np.identity(n)
    pivot = util.pivot(A, bh, list_zk, list_bhz, list_active_set, etaj, etajTy, cov, tn_mu, 'A')

    return pivot


if __name__ == '__main__':

    max_iteration = 150
    list_pivot = []

    for each_iter in range(max_iteration):
        np.random.seed(each_iter)
        print(each_iter)
        pivot = run()

        if pivot is not None:
            list_pivot.append(pivot)

    plt.rcParams.update({'font.size': 16})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_pivot))(grid), 'r-', linewidth=5, label='Pivot')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/uniform_pivot.png', dpi=100)
    plt.show()