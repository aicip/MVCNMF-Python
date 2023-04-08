import numpy as np


def conjugate(X, A, S, maxiter, U, mean_data, tao):
    """
    Projected conjugate gradient learning
    """

    maxiter = 1
    AtA = A.T @ A
    AtX = A.T @ X
    L, N = X.shape
    c, _ = S.shape

    cons = 0
    if L > N:
        cons = 1
        mean_data = mean_data.T @ np.ones((1, c))
        C = np.vstack((np.ones((1, c)), np.zeros((c - 1, c))))
        B = np.vstack((np.zeros((1, c - 1)), np.eye(c - 1)))
        Z = C + B @ U.T @ (S.T - mean_data)
        ZD = np.linalg.pinv(Z) @ B @ U.T
        detz2 = np.linalg.det(Z) ** 2

    if cons == 1:
        gradp = AtA @ S - AtX + tao * detz2 * ZD  # type: ignore
    else:
        gradp = AtA @ S - AtX

    conjp = gradp
    S = S - 0.001 * gradp

    for iter in range(maxiter):
        if cons == 1:
            Z = C + B @ U.T @ (S.T - mean_data)  # type: ignore
            ZD = np.linalg.pinv(Z) @ B @ U.T  # type: ignore
            detz2 = np.linalg.det(Z) ** 2
            grad = AtA @ S - AtX + tao * detz2 * ZD
        else:
            grad = AtA @ S - AtX

        beta = np.sum(grad * (grad - gradp), axis=1) / np.maximum(
            np.sum(gradp**2, axis=1), np.finfo(float).eps
        )

        conj = -grad + (beta * conjp.T).T

        AAd = AtA @ conj
        alpha = np.sum(conj * (-grad), axis=1) / np.maximum(
            np.sum(conj * AAd, axis=1), np.finfo(float).eps
        )

        S = S + (conj * alpha.reshape(-1, 1))

        gradp = grad
        conjp = conj

    return S
