import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from steepdescent import steep_descent as steepdescent
from conjugate import conjugate as conjugate


def print_summary(array, name):
    print("---------------------------")
    print(f"Size of {name}: {array.shape}")
    print(f"Minimum value: {np.min(array):.2f}")
    print(f"Maximum value: {np.max(array):.2f}")
    print(f"Mean value: {np.mean(array):.2f}")
    print(f"Standard deviation: {np.std(array):.2f}")
    print("---------------------------")


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
    A = Ainit.copy().astype('float64')
    S = Sinit.copy().astype('float64')

    # dimensions
    c = S.shape[0]  # number of endmembers
    N = S.shape[1]  # number of pixels

    PrinComp = PrinComp.T
    # precalculation for visualization
    if use_synthetic_data == 1:
        EM = UU.T @ Atrue  # low dimensional endmembers
        LowX = UU.T @ X  # low dimensional data

        E = np.vstack((np.ones((1, c)), 
                       PrinComp[:, : c - 1].T @ (Atrue - mean_data.T * np.ones((1, c)))))
        vol_t = (
            1 / factorial(c - 1) * abs(np.linalg.det(E))
        )  # the volume of true endmembers

    # PCA to calculate the volume of true EM
    vol = []

    # calculate volume of estimated A
    C = np.vstack((np.ones((1, c)), np.zeros((c - 1, c))))
    B = np.vstack((np.zeros((1, c - 1)), np.eye(c - 1)))
    Z = C + B @ (
        PrinComp[:, : c - 1].T
        @ (A - mean_data.T * np.ones((1, c)))
    )
    detz2 = np.linalg.det(Z) * np.linalg.det(Z)

    # one time draw
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

    # print_summary(detz2, "detz2")
    # print_summary(S, "S")
    # print_summary(B.T, "B.T")
    # print_summary(np.linalg.pinv(Z).T, "np.linalg.pinv(Z).T")
    # calculate initial gradient
    gradA = (
        A @ (S @ S.T)
        - X @ S.T
        + T * detz2 * PrinComp[:, : c - 1] @ B.T @ np.linalg.pinv(Z).T
    )
    # print_summary(gradA, "gradA")
    gradS = (A.T @ A) @ S - A.T @ X
    # print_summary(gradS, "gradS")
    initgrad = np.linalg.norm(np.vstack((gradA, gradS.T)), ord="fro")
    print(f"Init gradient norm {initgrad}")
    # tolA = max(0.001, tol) * initgrad
    # tolS = tolA

    # Calculate initial objective
    X[X<0.0] = 0.0
    # print_summary(A, "A")
    # print_summary(S, "S")
    # print_summary(X, "X")
    objhistory = 0.5 * np.sum(np.sum((X - A @ S) ** 2))
    print(f"Initial objective {objhistory}")
    objhistory = [objhistory, 0]
    Ahistory = []

    # count the number of sucessive increase of obj
    inc = 0
    inc0 = 0
    # flag = 0
    iter = 0

    while inc < 5 and inc0 < 20:
        # uphill or downhill
        if objhistory[-2] - objhistory[-1] > 0.0001:
            inc = 0
        elif objhistory[-1] - objhistory[-2] > 50:
            print(f"Diverge after {iter} iterations!")
            print(objhistory)
            print("objhistory[-1] - objhistory[-2] > 50")
            print(f"{objhistory[-1]} - {objhistory[-2]} > 50")
            print(f"{objhistory[-1] - objhistory[-2]} > 50")
            print("--------------------")
            print(f"[{iter}(final)]: {objhistory[-1]:.5f}\t", end="")
            break
        else:
            print("uphill")
            inc = inc + 1
            inc0 = inc0 + 1

        if iter < 5:
            inc = 0

        if iter == 0:
            objhistory[-1] = objhistory[-2]

        # stopping condition
        # projnorm = np.linalg.norm(
        #     np.hstack((gradA[gradA < 0 | A > 0], gradS[gradS < 0 | S > 0]))
        #     )

        if iter > maxiter:
            print("Max iter reached, stopping!")
            break

        # Show progress
        E = np.vstack(
            (
                np.ones((1, c)),
                PrinComp[:, : c - 1].T
                @ (A - mean_data.T @ np.ones((1, c))),
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
                Ahistory = np.column_stack([Ahistory, est])

            index = 0
            for i in range(3):
                for j in range(i + 1, 3):
                    ax = axes[index]
                    ax.plot(est[i, :], est[j, :], 'yo')  # estimation from nmf
                    index += 1

            plt.draw()
            plt.pause(0.001)  # pause to allow the figure to update

        # to consider the sum-to-one constraint
        tX = np.vstack((X, 20 * np.ones((1, N))))
        tA = np.vstack((A, 20 * np.ones((1, c))))
        # print_summary(tX, "tX-mvc")
        # print_summary(tA, "tA-mvc")
        # find S
        if type_alg_S == 1:  # conjugate gradient learning
            no_iter = 50
            S = conjugate(
                X, A, S, no_iter, PrinComp[:, : c - 1], mean_data, T
            )
        elif type_alg_S == 2:  # steepest descent
            tolS = 0.0001
            S, _, _ = steepdescent(
                tX, tA, S, tolS, 200, PrinComp[:, : c - 1], mean_data, T
            )

            # if iterS == 1:
            #     tolS = 0.1 * tolS

        # find A
        if type_alg_A == 1:  # conjugate gradient learning
            no_iter = 50
            A = conjugate(
                X.T, S.T, A.T, no_iter, PrinComp[:, : c - 1], mean_data, T
            )
            A = A.T
        elif type_alg_A == 2:  # steepest descent
            tolA = 0.0001
            A, _, _ = steepdescent(
                X.T, S.T, A.T, tolA, 100, PrinComp[:, : c - 1], mean_data, T
            )
            A = A.T
            # gradA = gradA.T

            # if iterA == 1:
            #     tolA = 0.1 * tolA
        # print_summary(A, "A-mvc")
        # print_summary(S, "S-mvc")
        # Calculate objective
        newobj = 0.5 * np.sum(np.sum((X - A @ S) ** 2))
        objhistory.append(newobj)

        iter = iter + 1

    return A, S
