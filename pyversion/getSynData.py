import numpy as np
from scipy.signal import convolve2d


def get_syn_data(A, win, pure):
    """
    Generate synthetic data.
    The spectra of simulated data is obtained from the
    USGS library "signatures"

    Input
        - A: matrix of reflectances
        - win: size of smoothing filter
        - pure: 0 - no pure pixels, 1 - exist pure pixel

    Output
        - mixed: generated synthetic mixed data
        - abf: actual abundance fractions

    The pure pixels can be removed by adding the following two lines
            ----Index = ceil(find(abf>0.8)/c);
            ----abf[:,Index] = 1/c*ones(c,1)*ones(0,length(Index));
    """

    band, c = A.shape
    dim = 64

    label = np.ones((dim // 8) ** 2, dtype=int)
    num = len(label) // c

    for i in range(c - 1):
        label[i * num: (i + 1) * num] = i + 2

    ridx = np.random.permutation(len(label))
    label = label[ridx].reshape(dim // 8, dim // 8)
    abf = np.zeros((dim, dim, c))
    img = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            for cls in range(c):
                if label[i // 8, j // 8] == cls + 1:
                    tmp = np.zeros(c)
                    tmp[cls] = 1
                    abf[i, j, :] = tmp
                    img[i, j] = c

    H = np.ones((win, win)) / (win * win)
    # img_fil = convolve2d(img, H, mode="same")

    for i in range(c):
        abf[:, :, i] = convolve2d(abf[:, :, i], H, mode="same")

    abf = abf[win // 2: -win // 2 + 1, win // 2: -win // 2 + 1, :]

    M, N, c = abf.shape
    abf = abf.reshape(M * N, c).T

    if pure == 0:
        index = np.ceil(np.argwhere(abf > 0.8) / c).flatten().astype(int)
        abf[:, index] = (1 / c) * np.ones((c, len(index)))

    mixed = (A @ abf).T.reshape(M, N, band)

    return mixed, abf
