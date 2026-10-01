import numpy as np

# Задача 1
def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0
    for i in range(len(ts1)):
        ed_dist += (ts1[i] - ts2[i])**2
    ed_dist = np.sqrt(ed_dist)


    return ed_dist

# Задача 2
def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size

    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    dtw_dist = D[n, n]
    return dtw_dist

# Задача 5
def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)

    # среднее и стандартное отклонение
    mu1 = np.mean(ts1)
    mu2 = np.mean(ts2)
    sigma1 = np.sqrt(np.sum(ts1**2) / n - mu1**2)
    sigma2 = np.sqrt(np.sum(ts2**2) / n - mu2**2)

    # скалярное произведение
    dot = np.dot(ts1, ts2)

    # нормализация заложена в формулу: вычитаем средние и делим на отклонения
    norm_ed_dist = np.sqrt(abs(2 * n * (1 - (dot - n * mu1 * mu2) / (n * sigma1 * sigma2))))

    return norm_ed_dist



