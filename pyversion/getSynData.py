import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve
# from mvcnmf import print_summary


def get_syn_data(A, win, pure):
    """
    Generate synthetic data.
    The spectra of simulated data is obtained from 
    the USGS library "signatures"

    Input
      - A: matrix of reflectances
      - win: size of smoothing filter
      - pure: 0 - no pure pixels, 1 - exist pure pixel

    Output
      - mixed: generated synthetic mixed data
      - abf: actual abundance fractions
    """

    band, c = A.shape
    dim = 64

    label = np.ones(((dim // 8) ** 2, 1))
    num = int(label.size / c)

    for i in range(c - 1):
        label[i * num:(i + 1) * num] = i + 2
    
    ridx = np.random.permutation(label.size)
    label = label[ridx].reshape(dim // 8, dim // 8)
    abf = np.zeros((dim, dim, c))
    img = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            for cls in range(c):
                if label[(i - 1) // 8, (j - 1) // 8] == cls + 1:
                    tmp = np.zeros(c)
                    tmp[cls] = 1
                    abf[i, j, :] = tmp
                    img[i, j] = c

    # low pass filter
    H = np.ones((win, win)) / (win * win)
    for i in range(c):
        abf[:, :, i] = convolve(abf[:, :, i], H)
    abf = abf[win // 2: -win // 2 + 1, win // 2: -win // 2 + 1, :]

    # generate mixtures
    M, N, c = abf.shape
    abf = abf.reshape((M * N, c)).T

    # remove pure pixels
    if pure == 0:
        rows, cols = np.where(abf > 0.8)
        index = np.unique(np.ceil(cols / c).astype(int))

        for idx in index:
            col_indices = np.where(np.ceil(cols / c).astype(int) == idx)[0]
            if idx < abf.shape[1]:
                abf[rows[col_indices], cols[col_indices]] = (1 / c) * np.ones(len(col_indices))


    # print_summary(A, "A-before")
    # print_summary(abf, "abf-before")
    mixed = (A @ abf).T.reshape(M, N, band)
    # print_summary(abf, "mixed-before")

    return mixed, abf

