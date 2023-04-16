import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from steepdescent import steep_descent as steepdescent
from conjugate import conjugate as conjugate


def mvcnmf_secord(
    X,
    Ainit,
    Sinit,
    Atrue,
    UU,
    PrinComp,
    mean_data,
    T,
    tol,
    maxiter,
    showflag,
    type_alg_S,
    type_alg_A,
    use_synthetic_data,
):
    A = Ainit.copy()
    S = Sinit.copy()

    c = S.shape[0]  # number of endmembers
    N = S.shape[1]  # number of pixels
    PrinComp = PrinComp.T
    if use_synthetic_data == 1:
        EM = UU.T @ Atrue  # low dimensional endmembers
        LowX = UU.T @ X  # low dimensional data

        _ones = np.ones((1, c))
        _PrinComp = PrinComp[:, : c - 1].T
        E = np.vstack((_ones, _PrinComp @ (Atrue - mean_data.T * _ones)))
        vol_t = (
            1 / factorial(c - 1) * abs(np.linalg.det(E))
        )  # the volume of true endmembers

    vol = []

    C = np.vstack((np.ones((1, c)), np.zeros((c - 1, c))))
    B = np.vstack((np.zeros((1, c - 1)), np.eye(c - 1)))
    Z = C + B @ (
        PrinComp[:, : c - 1].T
        @ (A - mean_data.T * np.ones((1, c)))
    )
    detz2 = np.linalg.det(Z) * np.linalg.det(Z)

    if showflag:
        startA = UU.T @ Ainit

        fig, axes = plt.subplots(2, 2)
        # Flatten the axes array to make it 
        # easier to index
        axes = axes.flatten()  # type: ignore
        index = 0
        for i in range(3):
            for j in range(i + 1, 3):
                ax = axes[index]
                ax.plot(LowX[i, ::6], LowX[j, ::6], "rx")
                ax.plot(startA[i, :], startA[j, :], "bo")
                ax.plot(EM[i, :], EM[j, :], "go", markerfacecolor="g")
                index += 1

        # Remove the unused subplot
        axes[-1].set_visible(False)

    gradA = (
        A @ (S @ S.T)
        - X @ S.T
        + T * detz2 * PrinComp[:, : c - 1] @ B.T @ np.linalg.pinv(Z).T
    )
    gradS = (A.T @ A) @ S - A.T @ X
    initgrad = np.linalg.norm(np.vstack((gradA, gradS.T)), ord="fro")
    print(f"Init gradient norm {initgrad}")
    tolA = max(0.001, tol) * initgrad
    tolS = tolA

    objhistory = 0.5 * np.sum(np.sum((X - A @ S) ** 2))
    objhistory = [objhistory, 0]
    Ahistory = []

    inc = 0
    inc0 = 0
    # flag = 0
    iter = 0

    while inc < 5 and inc0 < 20:
        if objhistory[-2] - objhistory[-1] > 0.0001:
            inc = 0
        elif objhistory[-1] - objhistory[-2] > 50:
            print(f"Diverge after {iter} iterations!")
            break
        else:
            print("uphill")
            inc = inc + 1
            inc0 = inc0 + 1

        if iter < 5:
            inc = 0

        if iter == 0:
            objhistory[-1] = objhistory[-2]

        # projnorm = np.linalg.norm(
        #     np.hstack((gradA[gradA < 0 | A > 0], gradS[gradS < 0 | S > 0]))
        # )

        if iter > maxiter:
            print("Max iter reached, stopping!")
            break

        E = np.vstack(
            (
                np.ones((1, c)),
                PrinComp[:, : c - 1].T
                @ (A - mean_data.T * np.ones((1, c))),
            )
        )
        vol_e = 1 / factorial(c - 1) * abs(np.linalg.det(E))
        print(f"[{iter}]: {objhistory[-1]:.5f}\t", end="")
        print(f"Temperature: {T} \t", end="")

        if use_synthetic_data == 1:
            print(f"Actual Vol.: {vol_t} \t Estimated Vol.: {vol_e}")
        else:
            print(f"Estimated Vol.: {vol_e}")

        vol.append(vol_e)

        if showflag:
            est = UU.T @ A

            if len(Ahistory) == 0:
                Ahistory = est
            else:
                Ahistory = np.hstack([Ahistory, est])

            index = 0
            for i in range(3):
                for j in range(i + 1, 3):
                    ax = axes[index]
                    ax.plot(est[i, :], est[j, :], 'yo')  # estimation from nmf
                    index += 1

            plt.draw()
            plt.pause(0.001)  # pause to allow the figure to update

        tX = np.vstack((X, 20 * np.ones((1, N))))
        tA = np.vstack((A, 20 * np.ones((1, c))))

        if type_alg_S == 1:
            no_iter = 50
            S = conjugate(
                X, A, S, no_iter, PrinComp[:, : c - 1], mean_data, T
            )
        elif type_alg_S == 2:
            tolS = 0.0001
            S, gradS, iterS = steepdescent(
                tX, tA, S, tolS, 200, PrinComp[:, : c - 1], mean_data, T
            )

            if iterS == 1:
                tolS = 0.1 * tolS

        if type_alg_A == 1:
            no_iter = 50
            A = conjugate(
                X.T, S.T, A.T, no_iter, PrinComp[:, : c - 1], mean_data, T
            )
            A = A.T
        elif type_alg_A == 2:
            tolA = 0.0001
            A, gradA, iterA = steepdescent(
                X.T, S.T, A.T, tolA, 100, PrinComp[:, : c - 1], mean_data, T
            )
            A = A.T
            gradA = gradA.T

            if iterA == 1:
                tolA = 0.1 * tolA

        newobj = 0.5 * np.sum(np.sum((X - A @ S) ** 2))
        objhistory.append(newobj)

        iter = iter + 1

    return A, S
