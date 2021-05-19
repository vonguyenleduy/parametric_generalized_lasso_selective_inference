import numpy as np


def generate(n, p, beta_vec):
    X = []
    y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        noise = np.random.normal(0, 1)
        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)

    return X, y


def generate_test(n, p, beta_vec):
    X = []
    y = []
    true_y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        true_y.append(yi)
        noise = np.random.normal(0, 1)
        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y


def generate_non_normal(n, p, beta_vec, type):
    X = []
    y = []
    true_y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        true_y.append(yi)

        noise = None

        if type == 'skew':
            from scipy.stats import skewnorm
            skew_coef = 10
            noise = skewnorm.rvs(a=skew_coef, loc=0, scale=1)
            true_mean = skewnorm.mean(a=skew_coef, loc=0, scale=1)
            true_std = skewnorm.std(a=skew_coef, loc=0, scale=1)
            noise = (noise - true_mean) / true_std

        elif type == 'laplace':
            from scipy.stats import laplace
            noise = laplace.rvs(loc=0, scale=1)
            true_mean = laplace.mean(loc=0, scale=1)
            true_std = laplace.std(loc=0, scale=1)
            noise = (noise - true_mean) / true_std

        elif type == 't20':
            from scipy.stats import t
            noise = t.rvs(df=20, loc=0, scale=1)
            true_mean = t.mean(df=20, loc=0, scale=1)
            true_std = t.std(df=20, loc=0, scale=1)
            noise = (noise - true_mean) / true_std

        elif type == 'sigma':
            noise = np.random.normal(0, 1)

        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y