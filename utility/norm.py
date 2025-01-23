import numpy as np

def rpkm(counts, lengths):
    N = np.sum(counts, axis=0)  # sum each column to get total reads per sample
    L = lengths
    C = counts
    normed = 1e9 * C / (N.values[np.newaxis, :] * L.values[:, np.newaxis])
    return normed

def tpm(counts, lengths):
    # Convert lengths to kilobases
    lengths_kb = lengths / 1e3
    # Calculate RPK
    rpk = counts / lengths_kb.values[:, np.newaxis]
    # Calculate scaling factor
    scaling_factor = np.sum(rpk, axis=0) / 1e6
    # Calculate TPM
    tpm = rpk / scaling_factor.values[np.newaxis, :]
    return tpm