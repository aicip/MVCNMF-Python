import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# from sklearn.linear_model import LassoLars
# from scipy.linalg import svdvals

# Custom libraries
from getSynData import get_syn_data as getSynData
from mvcnmf import mvcnmf_secord as mvcnmf
from vca import vca

# --- Parameters --- %
# input parameters

# Synthetic Data
use_synthetic_data = True
input_mat_name = "A.mat"
bands_mat_name = "BANDS.mat"

# Landsat Data
# use_synthetic_data = False
# input_mat_name = 'Landsat_separate_images_BR_R002.mat'
# input_mat_name = 'Landsat.mat'

# mvcnmf parameters
c = 5  # number of endmembers
SNR = 20  # dB
tol = 1e-6
maxiter = 150
T = 0.015
showflag = 1
verbose = True

# --- read data --- %
input_path = os.path.join("../inputs", input_mat_name)
# Load the list of variables in the .mat file
variables = [var for var in sio.whosmat(input_path)
             if var not in ['__header__', '__version__', '__globals__']]

if c == 1:
    raise ValueError("c must be greater than 1")

# start the timer
tic = time.time()

# loop through the variables
for i in range(len(variables)):
    print("#########################################")
    print(f"Processing {i+1}/{len(variables)} images")
    print("#########################################")
    variable_name = variables[i][0]
    # variable_name = "BR_R002_23KPR00_2014_01_09"
    # Load the first variable in the list
    loaded_variable = sio.loadmat(input_path)
    # Set variable A equal to the loaded variable
    A = loaded_variable[variable_name]
    if use_synthetic_data:
        # Load bands
        try:
            bands_path = os.path.join("../inputs", bands_mat_name)
            bands_mat = sio.loadmat(bands_path)
            BANDS = bands_mat["BANDS"].reshape(-1)
            A = A[BANDS, :c]
        except NameError:
            A = A[:, :c]
    # --- process --- %

    if use_synthetic_data:
        [synthetic, abf] = getSynData(A, 7, 0)
        # [M, N, D] = size(synthetic)
        M, N, D = synthetic.shape
        mixed = synthetic.reshape(M * N, D)
        print(f"Mixed has shape {mixed.shape}")
        # add noise
        variance = np.sum(mixed**2) / 10 ** (SNR / 10) / M / N / D
        n = np.sqrt(variance) * np.random.randn(D, M * N)
        mixed = mixed.T + n
        del n

        # remove noise
        UU, SS, VV = np.linalg.svd(mixed, full_matrices=False)
        Lowmixed = UU.T @ mixed
        mixed = UU @ Lowmixed
        EM = UU.T @ A
        # vca algorithm
        A_vca, EndIdx = vca(mixed, p=c, verbose=verbose)
    else:
        # load data
        M, N, D = A.shape
        mixed = A.reshape(M * N, D)
        mixed = mixed.T
        # create an empty var for UU
        UU = np.empty((0, 0))
        # vca algorithm
        A_vca, EndIdx = vca(mixed, p=c, verbose=verbose)
    # FCLS
    AA = np.vstack([1e-5 * A_vca, np.ones((1, A_vca.shape[1]))])
    s_fcls = np.zeros((A_vca.shape[1], M * N))

    for j in range(M * N):
        r = np.hstack([1e-5 * mixed[:, j], [1]])
        s_fcls[:, j] = np.linalg.lstsq(AA, r, rcond=None)[0]

    # use vca to initiate
    Ainit = A_vca
    sinit = s_fcls

    # PCA
    pca = PCA()
    PrinComp = pca.fit_transform(mixed.T)
    meanData = np.mean(PrinComp, axis=0)[:, np.newaxis].T

    # use conjugate gradient to find A can speed up the learning
    maxiter_str = f"{maxiter}"
    Aest, sest = mvcnmf(
        mixed,
        Ainit,
        sinit,
        A,
        UU,
        PrinComp,
        meanData,
        T,
        tol,
        maxiter,
        showflag,
        2,
        1,
        use_synthetic_data,
    )

    # visualize endmembers in scatterplots

    if showflag:
        d = 4
        Anmf = UU.T @ Aest
        
        fig, axes = plt.subplots(d - 2, d - 2)
        axes = np.ravel(axes)
        index = 0
        for i in range(d - 1):
            for j in range(i + 1, d - 1):
                ax = axes[index]
                ax.plot(Lowmixed[i, ::6], Lowmixed[j, ::6], 'rx')
                ax.plot(EM[i, :], EM[j, :], 'go', markerfacecolor='g')
                ax.plot(Anmf[i, :], Anmf[j, :], 'bo', markerfacecolor='b')
                index += 1

        plt.show()

    if use_synthetic_data == 1:
        # permute results
        CRD = np.corrcoef(np.hstack([A, Aest]))
        DD = np.abs(CRD[c : 2 * c, :c])
        perm_mtx = np.zeros((c, c))
        aux = np.zeros((c, 1))

        for i in range(c):
            ld, cd = np.unravel_index(np.argmax(DD), DD.shape)
            perm_mtx[ld, cd] = 1
            DD[:, cd] = aux.squeeze()
            DD[ld, :] = aux.squeeze().T
     
        Aest = Aest @ perm_mtx
        sest = (sest.T @ perm_mtx)
        Sest = np.reshape(sest, (M, N, c))
        sest = sest.T

    # show the estimations
    if showflag:
        fig, axs = plt.subplots(c, 4)

        for i in range(c):
            axs[i, 0].plot(A[:, i], "r", label="True endmembers")
            axs[i, 0].set_ylim(0, 1)

            if i == 0:
                axs[i, 0].set_title("True end-members")

            axs[i, 1].plot(Aest[:, i], "g", label="Estimated endmembers")
            axs[i, 1].legend()

            axs[i, 1].set_ylim(0, 1)

            if i == 0:
                axs[i, 1].set_title("Estimated end-members")

            axs[i, 2].imshow(abf[i, :].reshape(M, N))

            if i == 0:
                axs[i, 2].set_title("True abundance")

            axs[i, 3].imshow(sest[i, :].reshape(M, N))

            if i == 0:
                axs[i, 3].set_title("Estimated abundance")

        plt.show()

    # quantitative evaluation of spectral signature and abundance
    if use_synthetic_data:
        # rmse error of abundances
        E_rmse = np.sqrt(np.sum((abf - sest) ** 2) / (M * N * c))
        print(E_rmse)

        # the angle between abundances
        nabf = np.diag(abf @ abf.T)
        nsest = np.diag(sest @ sest.T)
        ang_beta = (
            180
            / np.pi
            * np.arccos(np.diag(abf @ sest.T) / np.sqrt(nabf * nsest))
        )
        E_aad = np.sqrt(np.mean(ang_beta**2))
        print(E_aad)

        # cross entropy between abundance
        E_entropy = np.sum(
            abf * np.log((abf + 1e-9) / (sest + 1e-9))
        ) + np.sum(sest * np.log((sest + 1e-9) / (abf + 1e-9)))
        E_aid = np.sqrt(np.mean(E_entropy**2))
        print(E_aid)

        # the angle between material signatures
        nA = np.diag(A.T @ A)
        nAest = np.diag(Aest.T @ Aest)
        ang_theta = (
            180 / np.pi * np.arccos(np.diag(A.T @ Aest) / np.sqrt(nA * nAest))
        )
        E_sad = np.sqrt(np.mean(ang_theta**2))
        print(E_sad)

        # the spectral information divergence
        pA = A / (np.sum(A, axis=0))
        qA = Aest / (np.sum(Aest, axis=0))
        qA = np.abs(qA)
        SID = np.sum(pA * np.log((pA + 1e-9) / (qA + 1e-9))) + np.sum(
            qA * np.log((qA + 1e-9) / (pA + 1e-9))
        )
        E_sid = np.sqrt(np.mean(SID**2))
        print(E_sid)

    # Save output
    # keep only 2 digits after the decimal point
    T_str = f"{T:.4f}"
    outputFileName = (
        f"../outputs/output_{variable_name}_max_iter{maxiter}_T{T_str}.mat"
    )

    if use_synthetic_data:
        data = {"Aest": Aest, "sest": sest, "E_rmse": E_rmse, 
                "E_aad": E_aad, "E_aid": E_aid, "E_sad": E_sad, 
                "E_sid": E_sid}
    else:
        data = {"Aest": Aest, "sest": sest}

    sio.savemat(outputFileName, data)
