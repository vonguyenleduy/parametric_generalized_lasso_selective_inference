import cvxpy as cp

def run(P, q, G, h, A, b, no_vars):
    x = cp.Variable(no_vars)

    q = q.flatten()
    h = h.flatten()
    b = b.flatten()

    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x),
                      [G @ x <= h,
                       A @ x == b])

    # prob.solve(verbose=True)
    prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, verbose=False)

    return x, prob