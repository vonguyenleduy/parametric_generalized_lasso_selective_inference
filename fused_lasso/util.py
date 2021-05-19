import numpy as np
from mpmath import mp

mp.dps = 500


def check_KKT(x, P, q, G, u, A, v):
    e_1 = np.dot(P, x)
    e_2 = q
    e_3 = np.dot(G.T, u)

    if A is None:
        e_4 = 0
    else:
        e_4 = np.dot(A.T, v)

    sum = e_1 + e_2 + e_3 + e_4

    print(np.around(sum, 10).flatten())


def check_zero(x):
    x_new = []

    threshold = 1e-10
    for e in x:
        if -threshold <= e <= threshold:
            x_new.append(0)
        else:
            x_new.append(e)

    return np.array(x_new)


def construct_A_XA_Ac_XAc_bhA(X, bh, p):
    A = []
    Ac = []
    bhA = []

    for j in range(p):
        bhj = bh[j]
        if bhj != 0:
            A.append(j)
            bhA.append(bhj)
        else:
            Ac.append(j)

    XA = X[:, A]
    XAc = X[:, Ac]
    bhA = np.array(bhA).reshape((len(A), 1))

    return A, XA, Ac, XAc, bhA


def construct_test_statistic(j, XA, y, A):
    ej = []
    for each_j in A:
        if j == each_j:
            ej.append(1)
        else:
            ej.append(0)

    ej = np.array(ej).reshape((len(A), 1))

    inv = np.linalg.inv(np.dot(XA.T, XA))
    XAinv = np.dot(XA, inv)
    etaj = np.dot(XAinv, ej)

    etajTy = np.dot(etaj.T, y)[0][0]

    return etaj, etajTy


def compute_a_b(y, etaj, n):
    sq_norm = (np.linalg.norm(etaj))**2

    e1 = np.identity(n) - (np.dot(etaj, etaj.T))/sq_norm
    a = np.dot(e1, y)

    b = etaj/sq_norm

    return a, b


def find_list_cp(beta, n):
    list_cp = [-1]
    current_cp_idx = -1
    for i in range(n):
        if np.abs(beta[i] - beta[current_cp_idx + 1]) > 1e-10:
            current_cp_idx = i - 1
            list_cp.append(current_cp_idx)

    list_cp.append(n - 1)

    return list_cp


def construct_s(bh):
    s = []

    for bhj in bh:
        if bhj != 0:
            s.append(np.sign(bhj))

    s = np.array(s)
    s = s.reshape((len(s), 1))
    return s


def pivot_with_constructed_interval(z_interval, etaj, etajTy, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None


def pivot(list_cp, path_zk, path_list_cp, etaj, etajTy, cov, tn_mu):

    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    z_interval = []

    for i in range(len(path_list_cp)):
        if np.array_equal(list_cp, path_list_cp[i]):
            z_interval.append([path_zk[i], path_zk[i + 1] - 1e-10])

    new_z_interval = []

    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval = new_z_interval

    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None


def p_value(list_cp, path_zk, path_list_cp, etaj, etajTy, cov):
    value = pivot(list_cp, path_zk, path_list_cp, etaj, etajTy, cov, 0)
    return 2 * min(1 - value, value)
