import numpy as np
import math
import pandas as pd
from scipy.stats import wasserstein_distance

def normalize_value(value, min_val=0, max_val=1):
    """
    Scale (Normalize) input value between min_val and max_val.

        Parameters
        ----------
        value : float
            Value to be normalized.
        min_val : slider
            Minimum bound of normalization.
        max_val : slider
            Maximum bound of normalization.

        Raises
        ------

        Returns
        -------
        val_norm: float
            Normalized value between min_val and max_val.

        See Also
        --------

        References
        ----------

        Examples
        --------
    """
    val_norm = (value - min_val) / (max_val - min_val)
    return val_norm


def jensen_shannon_distance(prob_dists):
    """
    Calculate the Jensen-Shannon distance for multiple probability distributions.

        Parameters
        ----------
        prob_dists : array-like
            Distribution (percentages) of labels for each local node (client).

        Raises
        ------

        Returns
        -------
        jsd_val: float
            Jensen-Shannon distance.

        See Also
        --------

        References
        ----------

        Examples
        --------
    """
    # Set weights to be uniform
    weight = 1 / len(prob_dists)
    js_left = np.zeros(len(prob_dists[0]))
    js_right = 0
    for pd in prob_dists:
        js_left += np.array(pd) * weight
        js_right += weight * entropy(pd, normalize=False)

    jsd_val = entropy(js_left, normalize=False) - js_right

    if len(prob_dists) > 2:
        jsd_val = normalize_value(jsd_val, min_val=0, max_val=math.log2(len(prob_dists)))
    return min(np.sqrt(jsd_val),1.0)


#
def entropy(prob_dist, normalize=True):
    """
    Calculate the entropy.

        Parameters
        ----------
        prob_dist : array-like
            Distribution (percentages) of labels for each local node (client).
        normalize : bool
            Flag to normalize the entropy.

        Raises
        ------

        Returns
        -------
        entropy_val: float
            Entropy.

        See Also
        --------

        References
        ----------

        Examples
        --------
    """
    entropy_val = -sum([p * math.log2(p) for p in prob_dist if p != 0])
    if normalize:
        max_entropy = math.log2(prob_dist.shape[0])
        return entropy_val / max_entropy
    return entropy_val


def get_spaced_colors(n):
    """
    Generate colors sufficiently spaced (visually different colors).

        Parameters
        ----------
        n : int
            Number of colors to generate.

        Raises
        ------

        Returns
        -------
        colors: list
            List of colors generated (hex format).

        See Also
        --------

        References
        ----------

        Examples
        --------
    """
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = ["#" + str(hex(i)[2:].zfill(6)) for i in range(0, max_value, interval)]
    return colors


def get_stratified_data(df, strat_var, strat_classes, strat_counts, random_state):

  df_strat = []

  for i in range(len(strat_classes)):
    if int(strat_counts[i]) == 0:
      df_filt = df[df[strat_var] == strat_classes[i]].sample(replace=True, n=1, random_state=random_state)
    else:
      df_filt = df[df[strat_var] == strat_classes[i]].sample(replace=True, n=int(strat_counts[i]), random_state=random_state)

    df_strat.append(df_filt)

  df_strat = pd.concat(df_strat, axis=0)

  return(df_strat)

def hellinger_distance(distributions):
    n = len(distributions)
    m = len(distributions[0])
    sqrt_d = np.sqrt(distributions)
    h = np.sum((sqrt_d[:, np.newaxis, :] - sqrt_d[np.newaxis, :, :]) ** 2, axis=2)
    return np.sqrt(np.sum(h) / (2 * n * (n - 1)))


def earth_movers_distance(D):
    D = np.array(D)
    # Calculate pairwise distances between distributions using EMD
    n = D.shape[0]
    emd_distances = np.zeros((n, n))
    for i, j in zip(*np.triu_indices(n, k=1)):
        emd_distances[i, j] = wasserstein_distance(D[i], D[j])
    emd_distances += emd_distances.T
    np.fill_diagonal(emd_distances, 0)

    # Calculate the average pairwise distance
    avg_distance = np.mean(emd_distances[np.triu_indices(n, k=1)])

    return (avg_distance)