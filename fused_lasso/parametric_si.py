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


def run_parametric_si(P, c, d, G, h, A, b, p, dim_x, z_threshold):
    zk = -z_threshold

    path_zk = [zk]
    path_list_cp = []

    while zk < z_threshold:

        qz = c + d * zk

        xz, probz = qp_solver.run(P, qz, G, h, A, b, dim_x)

        if probz.status != 'optimal':
            return None, None

        xz = xz.value
        uz = probz.constraints[0].dual_value
        vz = probz.constraints[1].dual_value

        beta_z = xz[0:p]
        list_cp_z = util.find_list_cp(beta_z, p)

        len_xz = len(xz)
        xz = xz.reshape((len_xz, 1))
        uz = uz.reshape((len(uz), 1))
        vz = vz.reshape((len(vz), 1))

        # util.check_KKT(xz, P, qz, G, uz, A, vz)

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

        dim_matrix = dim_x + len_Iz + A.shape[0]

        matrix = np.zeros((dim_matrix, dim_matrix))

        matrix[0:dim_x, 0:dim_x] = P
        matrix[0:dim_x, dim_x:dim_x + len_Iz] = GIz.T
        matrix[0:dim_x, dim_x + len_Iz:dim_matrix] = A.T

        matrix[dim_x:dim_x + len_Iz, 0:dim_x] = GIz
        matrix[dim_x + len_Iz:dim_matrix, 0:dim_x] = A

        matrix = np.linalg.inv(matrix)

        vector = np.concatenate((-d, np.zeros((len_Iz + A.shape[0], 1))), axis=0)

        dot_product = np.dot(matrix, vector)

        psi_z = dot_product[0:len_xz, :]
        gamma_z = dot_product[len_xz:len_xz + len_Iz, :]

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

        zk = zk + t_z + 0.0001

        if zk < z_threshold:
            path_zk.append(zk)
        else:
            path_zk.append(z_threshold)

        path_list_cp.append(list_cp_z)

    return path_zk, path_list_cp