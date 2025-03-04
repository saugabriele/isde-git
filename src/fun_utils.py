from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    n_samples = x.shape[0]
    n_tr = int(n_samples * tr_fraction)
    n_ts = n_samples - n_tr
    idx = np.linspace(0, n_samples - 1, num=n_samples, dtype=int)
    np.random.shuffle(idx)  # Inplace operation
    idx_tr = idx[:n_tr]
    idx_ts = idx[n_tr:n_tr + n_ts]

    x_tr = x[idx_tr, :]
    y_tr = y[idx_tr]

    x_ts = x[idx_ts, :]
    y_ts = y[idx_ts]
    return x_tr, y_tr, x_ts, y_ts
