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

    print(sum)


def check_sklearn(X, y, lamda, n):
    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=lamda / n, fit_intercept=False, normalize=False)
    clf.fit(X, y)
    bh = clf.coef_
    print(bh)


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


def compute_c_d(X, a, b, lamda, p):
    no_vars = 2 * p
    e_1 = lamda * np.ones((no_vars, 1))

    XTa = np.dot(X.T, a)
    XTb = np.dot(X.T, b)

    e_2 = np.zeros((no_vars, 1))
    e_2[0:p] = XTa
    e_2[p:2 * p] = -XTa

    e_3 = np.zeros((no_vars, 1))
    e_3[0:p] = XTb
    e_3[p:2 * p] = -XTb

    c = e_1 - e_2
    d = -e_3

    return c, d


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
    # print(tn_sigma)
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        # print(float(mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)))
        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None


def pivot(A, bh, list_zk, list_bhz, list_active_set, etaj, etajTy, cov, tn_mu, type):

    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    z_interval = []

    for i in range(len(list_active_set)):
        if type == 'As':
            if np.array_equal(np.sign(bh), np.sign(list_bhz[i])):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

        if type == 'A':
            if np.array_equal(A, list_active_set[i]):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

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


def p_value(A, bh, list_zk, list_bhz, list_active_set, etaj, etajTy, cov, type):
    value = pivot(A, bh, list_zk, list_bhz, list_active_set, etaj, etajTy, cov, 0, type)
    return min(1 - value, value)