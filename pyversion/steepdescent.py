import numpy as np


def steep_descent(X, A, S, tol, maxiter, U, mean_data, tao):
    """
    S, grad: output solution and gradient
    iter: #iterations used
    X, A: constant matrices
    tol: stopping tolerance
    maxiter: limit of iterations
    U, mean_data: principal components and data mean to calculate the volume
    tao: regularization parameter
    """
    grad = None
    L, N = X.shape
    c, N = S.shape

    # The constraint is only included when estimating A
    cons = 0
    if L > N:
        cons = 1

        # precalculation for volume constraint
        mean_data = mean_data.T @ np.ones((1, c))
        C = np.vstack((np.ones((1, c)), np.zeros((c - 1, c))))
        B = np.vstack((np.zeros((1, c - 1)), np.eye(c - 1)))

    # precalculation to reduce computational complexity
    AtX = A.T @ X
    AtA = A.T @ A

    alpha = 1
    beta = 0.1
    sigma = 0.01

    iter = None
    for iter in range(maxiter):
        # constraint on S^T
        if cons == 1:
            Z = C + B @ U.T @ (S.T - mean_data)  # type: ignore
            ZD = np.linalg.pinv(Z) @ B @ U.T  # type: ignore
            detz2 = np.linalg.det(Z) ** 2
            f = (np.sum(np.sum((X - A @ S) ** 2))
                 + tao * np.linalg.det(Z) ** 2)  # type: ignore

        # gradient with respective to S
        if cons == 1:
            grad = AtA @ S - AtX + tao * detz2 * ZD  # type: ignore
        else:
            grad = AtA @ S - AtX

        projgrad = np.linalg.norm(grad[(grad < 0) | (S > 0)])
        if projgrad < tol:
            break

        # search step size
        for inner_iter in range(50):
            Sn = np.maximum(S - alpha * grad, 0)
            d = Sn - S

            if cons == 1:
                fn = (
                    np.sum(np.sum((X - A @ Sn) ** 2))
                    + tao
                    * np.linalg.det(C + B @ U.T  # type: ignore
                                    @ (Sn.T - mean_data)) ** 2
                )
                suff_decr = (fn - f <=  # type: ignore
                             sigma * np.sum(np.sum(grad * d)))
            else:
                gradd = np.sum(np.sum(grad * d))
                dQd = np.sum(np.sum((AtA @ d) * d))
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0

            if inner_iter == 0:
                # the first iteration determines whether we should increase
                # or decrease alpha
                decr_alpha = not suff_decr
                Sp = S
            else:
                decr_alpha = None

            if decr_alpha:
                if suff_decr:
                    S = Sn
                    break
                else:
                    alpha = alpha * beta
            else:
                if not suff_decr or np.allclose(Sp, Sn):  # type: ignore
                    S = Sp  # type: ignore
                    break
                else:
                    alpha = alpha / beta
                    Sp = Sn
    if grad is None:
        raise ValueError("Gradient is None")
    return S, grad, iter
