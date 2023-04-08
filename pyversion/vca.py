import numpy as np
from scipy.sparse.linalg import svds


def vca(R, p, SNR=None, verbose=True):
    """
    Vertex Component Analysis

    Ae, indice, Rp = vca(R, p, SNR=None, verbose=True)

    Parameters
    ----------
    R : numpy array, shape (L, N)
        Matrix with dimensions L(channels) x N(pixels)
        Each pixel is a linear mixture of p endmembers
        signatures R = M x s, where s = gamma x alfa
        gamma is a illumination perturbation factor and
        alfa are the abundance fractions of each endmember.

    p : int
        Positive integer number of endmembers in the scene

    SNR : float, optional
        Signal to noise ratio (dB)

    verbose : bool, optional
        Print progress and additional information

    Returns
    -------
    Ae : numpy array
        Estimated mixing matrix (endmembers signatures)

    indice : numpy array
        Pixels that were chosen to be the most pure

    Rp : numpy array
        Data matrix R projected.
    """

    if R.size == 0:
        raise ValueError("There is no data")
    else:
        L, N = R.shape

    if p < 0 or p > L or p != int(p):
        raise ValueError(
            f"ENDMEMBERS parameter ({p}) must be integer between 1 and {L}"
        )

    if SNR is None:
        snr_input = 0
        r_m = np.mean(R, axis=1)[:, np.newaxis]
        R_m = np.tile(r_m, [1, N])
        R_o = R - R_m
        Ud, Sd, Vd = svds(R_o @ R_o.T / N, p)
        x_p = Ud.T @ R_o
        SNR = estimate_snr(R, r_m, x_p)
        if verbose:
            print(f"SNR estimated = {SNR:.2f}[dB]")
    else:
        snr_input = 1
        if verbose:
            print(f"input SNR = {SNR:.2f}[dB]")

    SNR_th = 15 + 10 * np.log10(p)

    if SNR < SNR_th:
        if verbose:
            print("... Select the projective projection")
        d = p - 1
        if snr_input == 0:
            Ud = Ud[:, :d]  # type: ignore
        else:
            r_m = np.mean(R, axis=1)[:, np.newaxis]
            R_m = np.tile(r_m, [1, N])
            R_o = R - R_m
            Ud, Sd, Vd = svds(R_o @ R_o.T / N, d)
            x_p = Ud.T @ R_o
        Rp = Ud @ x_p[:d, :] + np.tile(r_m, [1, N])  # type: ignore
        x = x_p[:d, :]  # type: ignore
        c = np.max(np.sum(x**2, axis=0)) ** 0.5
        y = np.vstack([x, c * np.ones((1, N))])
    else:
        if verbose:
            print("... Select projection to p-1")
        d = p
        Ud, Sd, Vd = svds(R @ R.T / N, d)
        x_p = Ud.T @ R
        Rp = Ud @ x_p[:d, :]
        x = Ud.T @ R
        u = np.mean(x, axis=1)[:, np.newaxis]
        y = x / np.tile(np.sum(x * np.tile(u, [1, N]), axis=0), [d, 1])

    indice = np.zeros(p, dtype=int)
    A = np.zeros((p, p))
    A[-1, 0] = 1

    for i in range(p):
        w = np.random.rand(p, 1)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / np.sqrt(np.sum(f**2))

        v = f.T @ y
        _, indice[i] = np.max(np.abs(v)), np.argmax(np.abs(v))
        A[:, i] = y[:, indice[i]]

    Ae = Rp[:, indice]

    return Ae, indice


def estimate_snr(R, r_m, x):
    L, N = R.shape
    p, _ = x.shape

    P_y = np.sum(R**2) / N
    P_x = np.sum(x**2) / N + r_m.T @ r_m
    snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est
