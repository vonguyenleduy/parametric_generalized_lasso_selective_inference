import numpy as np
import util
import qp_solver


def compute_quotient(numerator, denominator):
    if denominator == 0:
        return np.Inf

    quotient = numerator / denominator

    if quotient <= 0:
        return np.Inf

    return quotient


def run_parametric_si(X, P, G, h, p, c, d, z_threshold):
    zk = - z_threshold

    path_zk = [zk]
    path_bhz = []
    path_Az = []

    while zk < z_threshold:

        qz = c + d * zk
        xz, probz = qp_solver.solve_lasso(P, qz, G, h, 2 * p)

        xz = xz.value
        uz = probz.constraints[0].dual_value

        B_plus_z = xz[0:p]
        B_minus_z = xz[p:2 * p]

        beta_hat_z = B_plus_z - B_minus_z
        bhz = util.check_zero(beta_hat_z)

        Az, _, _, _, _ = util.construct_A_XA_Ac_XAc_bhA(X, bhz, p)


        len_xz = len(xz)
        xz = xz.reshape((len_xz, 1))
        uz = uz.reshape((len(uz), 1))

        Iz = []
        Icz = []

        for i in range(len(uz)):
            if uz[i][0] > 0:
                Iz.append(i)
            else:
                Icz.append(i)

        len_Iz = len(Iz)
        len_Icz = len(Icz)

        GIz = G[Iz, :]
        GIcz = G[Icz, :]

        matrix_1 = np.concatenate((P, GIz.T), axis=1)
        matrix_2 = np.concatenate((GIz, np.zeros((len_Iz, len_Iz))), axis=1)
        matrix = np.concatenate((matrix_1, matrix_2), axis=0)
        matrix = np.linalg.inv(matrix)

        vector = np.concatenate((-d, np.zeros((len_Iz, 1))), axis=0)

        dot_product = np.dot(matrix, vector)

        psi_z = dot_product[0:len_xz, :]
        gamma_z = dot_product[len_xz:, :]

        min1 = np.Inf
        min2 = np.Inf

        pre_numerator = np.dot(GIcz, xz) - h[Icz]
        pre_denominator = np.dot(GIcz, psi_z)

        for j in range(len_Icz):
            numerator = - pre_numerator[j][0]
            denominator = pre_denominator[j][0]

            quotient = compute_quotient(numerator, denominator)

            if quotient < min1:
                min1 = quotient

        pre_numerator = uz[Iz]
        pre_denominator = gamma_z

        for j in range(len_Iz):
            numerator = - pre_numerator[j][0]
            denominator = pre_denominator[j][0]

            quotient = compute_quotient(numerator, denominator)

            if quotient < min2:
                min2 = quotient

        t_z = min(min1, min2)

        zk = zk + t_z + 0.001

        if zk < z_threshold:
            path_zk.append(zk)
        else:
            path_zk.append(z_threshold)

        path_bhz.append(bhz)
        path_Az.append(Az)

    return path_zk, path_bhz, path_Az