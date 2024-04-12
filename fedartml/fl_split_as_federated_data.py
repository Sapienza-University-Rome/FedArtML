# Importing libraries
import numpy as np
import pandas as pd
from numpy.random import dirichlet
from fedartml.function_base import jensen_shannon_distance, hellinger_distance, earth_movers_distance


class SplitAsFederatedData:
    """
    Creates federated data from the provided centralized data (features and labels) to exemplify identically and
    non-identically distributed labels and features across the local nodes (clients). It allows one to select between
    two methods of data federation (percent_noniid and dirichlet). It works only for classification problems
    (labels as classes).

    Parameters
    ----------
    random_state : int
        Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
        output across multiple function calls.

    References
    ----------
    .. [1] (dirichlet) Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
           https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf
    .. [2] (percent_noniid) Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398). PMLR.
           https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    @staticmethod
    def percent_noniid_method(labels, local_nodes, pct_noniid=0, random_state=None):
        """
        Create a federated dataset divided per each local node (client) using the Percentage of Non-IID (pctg_noniid)
        method.

        Parameters
        ----------
        labels : array-like
            The target values (class labels in classification).
        local_nodes : int
            Number of local nodes (clients) used in the federated learning paradigm.
        pct_noniid : float
            Percentage (between o and 100) desired of non-IID-ness for the federated data.
        random_state : int
            Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
            output across multiple function calls.

        Returns
        -------
        pctg_distr : array-like
            Percentage (between 0 and 1) distribution of the classes for each local node (client).
        num_distr : array-like
            Numbers of distribution of the classes for each local node (client).
        idx_distr : array-like
            Indexes of examples (partition) taken for each local node (client).
        num_per_node : array-like
            Number of examples per each local node (client).

        References
        ----------
            .. [1] (percent_noniid) Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398).PMLR.
               https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf
        Examples
        --------
        """

        # Get number of examples in the noniid part
        n_noniid = int(len(labels) * (pct_noniid / 100))

        # sorted_labels = sorted(labels)
        sorted_labels = labels

        noniid_part_sample = list(sorted_labels[0:n_noniid])
        iid_part_sample = list(sorted_labels[n_noniid:len(labels)])

        uniq_class_noniid = np.unique(noniid_part_sample)
        n_class_per_node_noniid = len(uniq_class_noniid) // local_nodes

        pctg_distr = []
        num_distr = []
        idx_distr = []
        num_per_node = []

        n_ini = 0
        n_fin = n_class_per_node_noniid

        n_total_iid = len(sorted_labels) - len(noniid_part_sample)

        # Randomly assign each example to a local node (generate random numbers from 0 to local nodes)
        np.random.seed(random_state)
        rand_lnodes_iid = np.random.randint(0, local_nodes, size=n_total_iid)

        # Get data for each local node
        for i in range(local_nodes):
            # Get examples for noniid and iid parts
            aux_examples_node = [k for idx, k in enumerate(noniid_part_sample) if k in uniq_class_noniid[n_ini:n_fin]]
            idx_aux_examples_node = [idx for idx, k in enumerate(noniid_part_sample) if
                                     k in uniq_class_noniid[n_ini:n_fin]]
            sample_iid = [lab for idx, (lab, loc_nod) in enumerate(zip(iid_part_sample, rand_lnodes_iid)) if
                          loc_nod == i]
            idx_sample_iid = [idx + len(noniid_part_sample) for idx, (lab, loc_nod) in
                              enumerate(zip(iid_part_sample, rand_lnodes_iid)) if loc_nod == i]

            aux_examples_node = aux_examples_node + sample_iid
            idx_aux_examples_node = idx_aux_examples_node + idx_sample_iid

            # Get distribution of labels
            df_aux = pd.DataFrame(aux_examples_node, columns=['label']).label.value_counts().reset_index()
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(sorted_labels), columns=['index'])
            df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)

            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux_examples_node)
            idx_distr.append(idx_aux_examples_node)

            # Increase values to consider next iteration
            n_ini += n_class_per_node_noniid

            # Check if the iteration corresponds to the previous-to-last to add all the remaining values
            if i == (local_nodes - 2):
                n_fin = len(uniq_class_noniid) + 1
            else:
                n_fin += n_class_per_node_noniid

        return pctg_distr, num_distr, idx_distr, num_per_node

    @staticmethod
    def dirichlet_method(labels, local_nodes, alpha=1000, random_state=None):
        """
        Create a federated dataset divided per each local node (client) using the Dirichlet (dirichlet) method.

        Parameters
        ----------
        labels : array-like
            The target values (class labels in classification).
        local_nodes : int
            Number of local nodes (clients) used in the federated learning paradigm.
        alpha : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for
            the federated data.
        random_state : int
            Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
            output across multiple function calls.

        Returns
        -------
        pctg_distr : array-like
            Percentage (between 0 and 1) distribution of the classes for each local node (client).
        num_distr : array-like
            Numbers of distribution of the classes for each local node (client).
        idx_distr : array-like
            Indexes of examples (partition) taken for each local node (client).
        num_per_node : array-like
            Number of examples per each local node (client).

        References
        ----------
            .. [1] (dirichlet) Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
                https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf
        """
        # https://github.com/Xtra-Computing/NIID-Bench/blob/main/partition.py
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        labels = np.array(labels)

        min_size = 0
        num_classes = len(np.unique(labels))
        N = labels.shape[0]
        random_state_loop = random_state

        while min_size < 10:
            idx_batch = [[] for _ in range(local_nodes)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.seed(random_state_loop)
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / local_nodes) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                if random_state is not None:
                    random_state_loop += 100

        pctg_distr = []
        num_distr = []
        idx_distr = []
        num_per_node = []

        random_state_loop = random_state
        for j in range(local_nodes):
            np.random.seed(random_state_loop)
            np.random.shuffle(idx_batch[j])

            # Get examples for each batch from labels
            aux_examples_node = labels[idx_batch[j]]
            # Get distribution of labels
            df_aux = pd.DataFrame(aux_examples_node, columns=['label']).label.value_counts().reset_index()
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(labels), columns=['index'])
            df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)

            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux_examples_node)
            idx_distr.append(idx_batch[j])
            if random_state is not None:
                random_state_loop += 100

        return pctg_distr, num_distr, idx_distr, num_per_node

    @staticmethod
    def add_gaussian_noise(feat, mu=0, sigma=0, client_id=0, local_nodes=4, random_state=None):
        """
        Add Gaussian random noise to given features.

        Parameters
        ----------
        feat : array-like
            List of numpy arrays (or pandas dataframe) with images (i.e. features).
        mu : float
            Mean (“centre”) of the Gaussian distribution.
        sigma : float
            Standard deviation (noise) of the Gaussian distribution. Must be non-negative.
        client_id : int
            Identification or number of the client to add the noise.
        local_nodes : int
            Number of local nodes (clients) used in the federated learning paradigm.
        random_state: int
            Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
            output across multiple function calls.

        Returns
        -------
        feat : array-like
            List of numpy arrays (or pandas dataframe) with images (i.e. features) with the random noise applied.

        References
        ----------
            .. [1] (gaussian noise) Li, Q., Diao, Y., Chen, Q., & He, B. (2022, May). Federated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) (pp. 965-978). IEEE.
        """

        noise_level = sigma * client_id / local_nodes
        np.random.seed(random_state)
        noise = np.random.normal(mu, noise_level, feat.shape)
        feat = feat + noise
        return feat

    @staticmethod
    def calculate_bins_range(column, sigma_noise, n_bins):
        min_val = column.min()
        max_val = column.max()
        # At 4 deviations from the mean the data will keep almost at 100%
        bins_range = np.array(
            np.linspace(min_val - 4 * sigma_noise, max_val + 4 * sigma_noise, num=n_bins, endpoint=True))

        return bins_range

    @staticmethod
    def create_histogram(flat_input, bins):
        """
        Create histogram and bins from given flatted features.

        Parameters
        ----------
        flat_input : array-like (flatten)
            List of numpy arrays (or pandas dataframe) with images (i.e. features) flatten.
        bins : int
            Number of bins to use in the histogram.

        Returns
        -------
        histogram : array-like
            The values of the histogram. Normalized to sum up to 1.
        bin_edges : array-like
            The bin edges.

        References
        ----------

        """
        histogram, bin_edges = np.histogram(flat_input, bins=bins)
        histogram = histogram / flat_input.shape[0]

        return histogram

    @staticmethod
    def dirichlet_method_quant_skew(labels, local_nodes, alpha=1000, random_state=None, method="no-quant-skew"):
        """
        Create a federated dataset divided per each local node (client) using the Dirichlet (dirichlet) method to evaluate quantity skew.

        Parameters
        ----------
        labels : array-like
            The target values (class labels in classification).
        local_nodes : int
            Number of local nodes (clients) used in the federated learning paradigm.
        alpha : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for
            the federated data.
        random_state : int
            Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
            output across multiple function calls.
        method : str
            Method to create the federated data based on quantity skew. Possible options: "no-quant-skew"(default), "dirichlet", "minsize-dirichlet"

        Returns
        -------
        pctg_distr : array-like
            Percentage (between 0 and 1) distribution of the classes for each local node (client).
        num_distr : array-like
            Numbers of distribution of the classes for each local node (client).
        idx_distr : array-like
            Indexes of examples (partition) taken for each local node (client).
        num_per_node : array-like
            Number of examples per each local node (client).

        References
        ----------
            .. [1] (dirichlet) Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
                https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf
        """
        # https://github.com/Xtra-Computing/NIID-Bench/blob/main/partition.py
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        labels = np.array(labels)

        min_size = 0

        N = labels.shape[0]
        random_state_loop = random_state

        np.random.seed(random_state_loop)
        idxs = np.random.permutation(N)

        min_require_size = len(np.unique(labels)) * 3

        if method == "dirichlet":
            while min_size < min_require_size:
                proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
                proportions = proportions / proportions.sum()
                min_size = np.min(proportions * len(idxs))
        elif method == "minsize-dirichlet":
            proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
            proportions = proportions / proportions.sum()
            proportions = [(min_require_size + 1) / len(idxs) if i < + (min_require_size + 1) / len(idxs) else i for i
                           in proportions]
            proportions = [i / sum(proportions) for i in proportions]

        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]

        idx_batch = np.split(idxs, proportions)

        idx_batch = [list(value) for value in idx_batch]

        pctg_distr = []
        num_distr = []
        idx_distr = []
        num_per_node = []

        random_state_loop = random_state
        for j in range(local_nodes):
            np.random.seed(random_state_loop)
            np.random.shuffle(idx_batch[j])

            # Get examples for each batch from labels
            aux_examples_node = labels[idx_batch[j]]
            # Get distribution of labels
            df_aux = pd.DataFrame(aux_examples_node, columns=['label']).label.value_counts().reset_index()
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(labels), columns=['index'])
            df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)

            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux_examples_node)
            idx_distr.append(idx_batch[j])
            if random_state is not None:
                random_state_loop += 100

        return pctg_distr, num_distr, idx_distr, num_per_node

    @staticmethod
    def st_dirichlet_method(labels, local_nodes, alpha=1000, random_state=None, st_variable=None):
        """
        Create a federated dataset divided per each local node (client) using the Dirichlet (dirichlet) method.

        Parameters
        ----------
        labels : array-like
            The target values (class labels in classification).
        local_nodes : int
            Number of local nodes (clients) used in the federated learning paradigm.
        alpha : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for
            the federated data.
        random_state : int
            Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
            output across multiple function calls.
        st_variable : array-like
            The spatio-temporal variable from the centralized data.
        Returns
        -------
        pctg_distr : array-like
            Percentage (between 0 and 1) distribution of the classes for each local node (client).
        num_distr : array-like
            Numbers of distribution of the classes for each local node (client).
        idx_distr : array-like
            Indexes of examples (partition) taken for each local node (client).
        num_per_node : array-like
            Number of examples per each local node (client).
        pctg_distr_st_var : array-like
            Percentage (between 0 and 1) distribution of the spatio-temporal variable's categories for each local node (client).
        References
        ----------

        """
        st_variable = np.array(st_variable)
        labels = np.array(labels)
        # print(st_variable)
        # print(st_variable)
        min_size = 0
        num_categ = len(np.unique(st_variable))
        N = st_variable.shape[0]
        random_state_loop = random_state

        while min_size < 10:
            idx_batch = [[] for _ in range(local_nodes)]
            for k in range(num_categ):
                idx_k = np.where(st_variable == k)[0]
                np.random.seed(random_state_loop)
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / local_nodes) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                if random_state is not None:
                    random_state_loop += 100

        pctg_distr = []
        num_distr = []
        idx_distr = []
        num_per_node = []

        pctg_distr_st_var = []

        random_state_loop = random_state
        for j in range(local_nodes):
            np.random.seed(random_state_loop)
            np.random.shuffle(idx_batch[j])

            aux_examples_node = labels[idx_batch[j]]
            # Get distribution of labels
            df_aux = pd.DataFrame(aux_examples_node, columns=['label']).label.value_counts().reset_index()
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(labels), columns=['index'])
            df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)

            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux_examples_node)
            idx_distr.append(idx_batch[j])

            # Get spatio-temporal variable distribution per node
            aux_examples_node = st_variable[idx_batch[j]]
            # Get distribution of labels
            df_aux = pd.DataFrame(aux_examples_node, columns=['st_var']).st_var.value_counts().reset_index()
            df_aux.columns = ['index', 'st_var']
            df_node = pd.DataFrame(np.unique(st_variable), columns=['index'])
            df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)

            df_node['perc'] = df_node.st_var / sum(df_node.st_var)

            pctg_distr_st_var.append(list(df_node.perc))

            if random_state is not None:
                random_state_loop += 100

        return pctg_distr, num_distr, idx_distr, num_per_node, pctg_distr_st_var

    def create_clients(self, image_list, label_list, num_clients=4, prefix_cli='client', method="dirichlet",
                       alpha=1000, percent_noniid=0, sigma_noise=0, bins='n_samples', feat_sample_rate=0.1,
                       feat_skew_method="gaussian-noise", alpha_feat_split=1000, idx_feat='feat-mean',
                       feat_quantile=20, quant_skew_method="no-quant-skew", alpha_quant_split=1000,
                       spa_temp_skew_method="no-spatemp-skew", alpha_spa_temp=1000, spa_temp_var=None):
        """
        Create a federated dataset divided per each local node (client) using the desired method (percent_noniid or dirichlet). It works only for classification problems (labels as classes) with quantitaive (numeric) features.

        Parameters
        ----------
        image_list : array-like
            List of numpy arrays (or pandas dataframe) with images (i.e. features) from the centralized data.
        label_list : array-like
            The target values (class labels in classification) from the centralized data.
        num_clients : int
            Number of local nodes (clients) used in the federated learning paradigm.
        prefix_cli : str
            The clients' name prefix, e.g., client_1, client_2, etc.
        method : string
            Method to create the federated data based on label skew. Possible options: "percent_noniid"(default), "dirichlet", "no-label-skew"
        alpha : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for the labels of the federated data.
        percent_noniid : float
            Percentage (between o and 100) desired of non-IID-ness for the labels of the federated data.
        sigma_noise : float
            Noise (sigma parameter of Gaussian distro) to be added to the features. Applicable only for feat_skew_method="gaussian-noise".
        bins : int or str
            Number of bins used to create histogram of features to check feature skew. It can be the word 'n_samples' or the integer number of bins to use. If 'n_samples'(default) is selected, then it is set as the number values of the image_list (examples). Applicable only for feat_skew_method="gaussian-noise".
        feat_sample_rate : float
            Proportion (between 0 and 1) to be sampled from features. This parameter is useful when dealing with datasets with many features (i.e. images). Applicable only for feat_skew_method="gaussian-noise".
        feat_skew_method : str
            Method to create the federated data based on feature skew. Possible options: "gaussian-noise"(default), "hist-dirichlet"
        alpha_feat_split : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for the features of the federated data. Applicable only for feat_skew_method="hist-dirichlet".
        idx_feat : int or str
            Position (idx) of feature used to simulate feature skew. It can be the word 'feat-mean' or the integer number of the position to use. If 'feat-mean'(default) is selected, then the mean of all the features is computed as representative of the features. Applicable only for feat_skew_method="hist-dirichlet".
        feat_quantile : int
            Number quantiles to use in the feature skew simulation. 20 for ventiles (default), 10 for deciles, 4 for quartiles, etc. Applicable only for feat_skew_method="hist-dirichlet".
        quant_skew_method : str
            Method to create the federated data based on quantity skew. Possible options: "no-quant-skew"(default), "dirichlet", "minsize-dirichlet"
        alpha_quant_split : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for the quantity skew of the federated data. Applicable only for quant_skew_method="dirichlet".
        spa_temp_skew_method : str
            Method to create the federated data based on spatio-temporal skew. Possible options: "no-spatemp-skew"(default), "st-dirichlet"
        alpha_spa_temp : float
            Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for the spatio-temporal skew of the federated data. Applicable only for spa_temp_skew_method="st-dirichlet".
        spa_temp_var : array-like
            The spatio-temporal variable from the centralized data. Applicable only for spa_temp_skew_method="st-dirichlet".
        Returns
        -------
        fed_data : dict
            Contains features (images) and labels for each local node (client) after federating the data. Includes "with_class_completion" and "without_class_completion" cases.
        ids_list_fed_data : array-like
            Indexes of examples (partition) taken for each local node (client).
        num_missing_classes : array-like
            Number of missing classes per each local node when creating the federated dataset
        distances : dict
            Distances calculated while measuring heterogeneity (non-IID-ness) of the label's distribution among clients. Includes "with_class_completion" and "without_class_completion" cases.
        spatemp_fed_data : dict
            Contains categories of the spatio-temporal variable for each local node (client) after federating the data. It is generated only when spa_temp_skew_method = "st-dirichlet".

        Note: When creating federated data and setting heterogeneous distributions (i.e. high values of percent_noniid or small values of alpha), it is more likely the clients hold examples from only one class. Then, two cases (for labels and features) are returned as output for fed_data and distances:
            - "with_class_completion": In this case, the clients are completed with one (random) example of each missing class for each client to have all the label's classes.
            - "without_class_completion": In this case, the clients are NOT completed with one (random) example of each missing class. Consequently, summing the number of examples of each client results in the same number of total examples (number of rows in image_list).

        References
        ----------
            .. [1] (dirichlet) Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning0
               https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf
            .. [2] (percent_noniid) Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398).PMLR.
               https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf
            .. [3] (gaussian noise) Li, Q., Diao, Y., Chen, Q., & He, B. (2022, May). Federated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) (pp. 965-978). IEEE.
        Examples
        --------
        >>> from fedartml import SplitAsFederatedData
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_federater = SplitAsFederatedData(random_state=0)
        >>>
        >>> # Using percent_noniid method
        >>> clients_glob, list_ids_sampled, miss_class_per_node, distances =
        >>>     my_federater.create_clients(image_list=train_X, label_list=train_y, num_clients=4,
        >>>     prefix_cli='Local_node',method="percent_noniid", percent_noniid=0)
        >>>
        >>> # Using dirichlet method
        >>> clients_glob, list_ids_sampled, miss_class_per_node, distances =
        >>>     my_federater.create_clients(image_list=train_X, label_list=train_y, num_clients=4,
        >>>     prefix_cli='Local_node',method="dirichlet", alpha=1000)
        """
        # create a list of client names
        client_names = ['{}_{}'.format(prefix_cli, i + 1) for i in range(num_clients)]

        # Zip the data as list
        data = list(zip(image_list, label_list))

        if (method == "percent_noniid" or method == "dirichlet") and feat_skew_method == "hist-dirichlet":
            raise ValueError(
                "The hist-dirichlet method can't be used simultaneously with dirichlet nor percent_noniid label skew methods. If you intent to use hist-dirichlet use method == 'no-label-skew'")
        elif (quant_skew_method == "dirichlet") and feat_skew_method == "hist-dirichlet":
            raise ValueError(
                "The hist-dirichlet method can't be used simultaneously with dirichlet quantity skew methods. If you intent to use hist-dirichlet use quant_skew_method == 'no-quant-skew'")
        if (method == "percent_noniid" or method == "dirichlet") and quant_skew_method == "dirichlet":
            raise ValueError(
                "The dirichlet (for quantity skew) method can't be used simultaneously with dirichlet nor percent_noniid label skew methods. If you intent to use dirichlet (for quantity skew) use method == 'no-label-skew'")
        if (
                method == "no-label-skew" and quant_skew_method == "no-quant-skew" and spa_temp_skew_method == "no-spatemp-skew") and feat_skew_method == "gaussian-noise":
            raise ValueError(
                "When using Gaussian Noise (for feature skew) either 'method', 'quant_skew_method' or 'temp_skew_method' should be different to 'no-label-skew','no-quant-skew' or 'no-spatemp-skew', respectively.")
        elif feat_skew_method == "gaussian-noise":
            num_missing_classes = []
            # Set list to append labels and features for each client
            shards_no_completion = []
            shards_with_completion = []

            random_state_loop = self.random_state

            # List to append the position of the recordings extracted
            ids_list_no_completion = []
            ids_list_with_completion = []

            if method == "dirichlet":
                lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
                    self.dirichlet_method(labels=label_list, local_nodes=num_clients, alpha=alpha,
                                          random_state=random_state_loop)
            elif method == "percent_noniid":
                lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
                    self.percent_noniid_method(labels=label_list, local_nodes=num_clients, pct_noniid=percent_noniid,
                                               random_state=random_state_loop)
            elif quant_skew_method == "dirichlet":
                lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
                    self.dirichlet_method_quant_skew(labels=label_list, local_nodes=num_clients,
                                                     alpha=alpha_quant_split, random_state=random_state_loop,
                                                     method=quant_skew_method)
            elif quant_skew_method == "minsize-dirichlet":
                lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
                    self.dirichlet_method_quant_skew(labels=label_list, local_nodes=num_clients,
                                                     alpha=alpha_quant_split, random_state=random_state_loop,
                                                     method=quant_skew_method)
            elif spa_temp_skew_method == "st-dirichlet":
                lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node, \
                st_var_dist_cli_pctg = self.st_dirichlet_method(labels=label_list, local_nodes=num_clients,
                                                                alpha=alpha_spa_temp, random_state=random_state_loop,
                                                                st_variable=spa_temp_var)

            elif method not in ['percent_noniid', 'dirichlet', 'no-label-skew']:
                raise ValueError("Method '" + method +
                                 "' not implemented. Available label skew methods are: ['percent_noniid', "
                                 "'dirichlet', 'no-label-skew'].")
            else:
                raise ValueError("Method '" + quant_skew_method +
                                 "' not implemented. Available quantity skew methods are: ['dirichlet', "
                                 "'minsize-dirichlet', 'no-quant-skew']")
            # Calculate Jensen-Shannon distance
            JS_dist = jensen_shannon_distance(lbl_distro_clients_pctg)
            # Calculate Hellinger distance
            H_dist = hellinger_distance(lbl_distro_clients_pctg)
            # Calculate Earth Mover’s distance
            emd_dist = earth_movers_distance(lbl_distro_clients_pctg)

            distances = {'without_class_completion': {'jensen-shannon': JS_dist, 'hellinger': H_dist,
                                                      'earth-movers': emd_dist}}

            data_df = pd.DataFrame(data)
            data_df.columns = [*data_df.columns[:-1], 'class']

            if spa_temp_skew_method == "st-dirichlet":
                spatem_df = pd.DataFrame(spa_temp_var, columns=['spatemp_var'])

            fed_data = {}
            ids_list_fed_data = {}
            pctg_distr = []
            dist_hist_no_completion = []
            dist_hist_with_completion = []
            st_var_cli_list = []
            spatemp_fed_data = {}

            # Define number of bins for histogram
            if bins == 'n_samples':
                n_bins = np.array(image_list).shape[0]
            else:
                n_bins = bins

            shape_x = np.array(np.array(image_list).shape)

            # Select randomly some features for measuring feature skew
            feat_sample_size = max(int(feat_sample_rate * np.prod(shape_x[1:])), 1)
            np.random.seed(self.random_state)
            idx_samp_feat = np.random.choice(np.arange(np.prod(shape_x[1:])), size=feat_sample_size, replace=False)
            features = np.array(image_list).reshape((shape_x[0], np.prod(shape_x[1:])))[:, idx_samp_feat]

            # Calculate bins_range with noise
            bins_range = np.apply_along_axis(self.calculate_bins_range, axis=0, arr=features, sigma_noise=sigma_noise,
                                             n_bins=n_bins)

            for i in range(num_clients):

                X = data_df.iloc[lbl_distro_clients_idx[i], 0].values
                y = data_df.iloc[lbl_distro_clients_idx[i], 1].values

                if isinstance(X[0], list):
                    X = np.array(X.tolist())

                if sigma_noise > 0:
                    X = self.add_gaussian_noise(feat=X, sigma=sigma_noise, client_id=i + 1, local_nodes=num_clients,
                                                random_state=random_state_loop)
                    X = np.array(X.tolist())

                    # flattenX = np.concatenate([np.ravel(X[j]) for j in range(X.shape[0])])
                    # Select randomly some features for measuring feature skew
                    shape_x = np.array(X.shape)
                    features = X.reshape((shape_x[0], np.prod(shape_x[1:])))[:, idx_samp_feat]

                    # Calculate histograms for each column
                    histograms = np.array([self.create_histogram(column, bins) for column, bins in zip(features.T,
                                                                                                       bins_range.T)])
                    del features
                else:
                    histograms = np.zeros((features.shape[1], 20))

                dist_hist_no_completion.append(list(histograms))

                del histograms

                if i == (num_clients - 1):
                    # Reshape to make calculations per client
                    dist_hist_no_completion = np.transpose(np.array(dist_hist_no_completion), (1, 0, 2)).tolist()
                    dists = np.array(list(map(jensen_shannon_distance, dist_hist_no_completion)))
                    JS_dist_feat = np.mean(dists)
                    dists = np.array(list(map(hellinger_distance, dist_hist_no_completion)))
                    H_dist_feat = np.mean(dists)
                    dists = np.array(list(map(earth_movers_distance, dist_hist_no_completion)))
                    emd_dist_feat = np.mean(dists)

                    del dist_hist_no_completion

                X = X.tolist()
                y = y.tolist()

                # Get the index (name) of the recordings sampled
                ids_list_no_completion.append(lbl_distro_clients_idx[i])

                shards_no_completion.append(list(zip(X, y)))

                if spa_temp_skew_method == "st-dirichlet":
                    st_var_cli = spatem_df.iloc[lbl_distro_clients_idx[i], 0].values.tolist()
                    st_var_cli_list.append(list(st_var_cli))

                # Add missing classes when sampling (mainly for extreme case percent iid = 100)
                # diff_classes = list(set(label_list) - set(lbl_distro_clients_num[i]))
                diff_classes = list(set(label_list) - set(y))
                num_diff_classes = len(diff_classes)
                num_missing_classes.append(num_diff_classes)

                if num_diff_classes > 0:
                    for k in diff_classes:
                        vals = [idx for idx, y in enumerate(label_list) if y == k][0]
                        lbl_distro_clients_idx[i] = lbl_distro_clients_idx[i] + [vals]

                X = data_df.iloc[lbl_distro_clients_idx[i], 0].values
                y = data_df.iloc[lbl_distro_clients_idx[i], 1].values

                if isinstance(X[0], list):
                    X = np.array(X.tolist())

                if sigma_noise > 0:
                    X = self.add_gaussian_noise(feat=X, sigma=sigma_noise, client_id=i + 1, local_nodes=num_clients,
                                                random_state=random_state_loop)
                    X = np.array(X.tolist())
                    # flattenX = np.concatenate([np.ravel(X[j]) for j in range(X.shape[0])])
                    # Select randomly some features for measuring feature skew
                    shape_x = np.array(X.shape)
                    features = X.reshape((shape_x[0], np.prod(shape_x[1:])))[:, idx_samp_feat]

                    # Calculate histograms for each column
                    histograms = np.array([self.create_histogram(column, bins) for column, bins in zip(features.T,
                                                                                                       bins_range.T)])
                    del features
                else:
                    histograms = np.zeros((features.shape[1], 20))

                dist_hist_with_completion.append(list(histograms))
                del histograms

                X = X.tolist()
                y = y.tolist()

                # Get distribution of labels
                df_aux = pd.DataFrame(y, columns=['label']).label.value_counts().reset_index()
                df_aux.columns = ['index', 'label']
                df_node = pd.DataFrame(np.unique(label_list), columns=['index'])
                df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
                df_node['perc'] = df_node.label / sum(df_node.label)

                pctg_distr.append(list(df_node.perc))

                # Get the index (name) of the recordings sampled
                ids_list_with_completion.append(lbl_distro_clients_idx[i])

                shards_with_completion.append(list(zip(X, y)))
                if self.random_state is not None:
                    random_state_loop += self.random_state + 100

            # Reshape to make calculations per client
            dist_hist_with_completion = np.transpose(np.array(dist_hist_with_completion), (1, 0, 2)).tolist()

            # Add elements to dictionary of federated data
            fed_data['with_class_completion'] = \
                {client_names[i]: shards_with_completion[i] for i in range(len(client_names))}
            fed_data['without_class_completion'] = \
                {client_names[i]: shards_no_completion[i] for i in range(len(client_names))}

            # Add elements to dictionary of ids list of federated data
            ids_list_fed_data['with_class_completion'] = ids_list_with_completion
            ids_list_fed_data['without_class_completion'] = ids_list_no_completion

            # Calculate Jensen-Shannon distance for labels
            JS_dist = jensen_shannon_distance(pctg_distr)
            # Calculate Hellinger distance for labels
            HD_dist = hellinger_distance(pctg_distr)
            # Calculate Earth Mover’s distance for labels
            emd_dist = earth_movers_distance(pctg_distr)
            distances['with_class_completion'] = {'jensen-shannon': JS_dist, 'hellinger': HD_dist,
                                                  'earth-movers': emd_dist}

            distances['without_class_completion_feat'] = {'jensen-shannon': JS_dist_feat, 'hellinger': H_dist_feat,
                                                          'earth-movers': emd_dist_feat}

            dists = np.array(list(map(jensen_shannon_distance, dist_hist_with_completion)))
            JS_dist_feat = np.mean(dists)
            dists = np.array(list(map(hellinger_distance, dist_hist_with_completion)))
            H_dist_feat = np.mean(dists)
            dists = np.array(list(map(earth_movers_distance, dist_hist_with_completion)))
            emd_dist_feat = np.mean(dists)

            distances['with_class_completion_feat'] = {'jensen-shannon': JS_dist_feat, 'hellinger': H_dist_feat,
                                                      'earth-movers': emd_dist_feat}

            if spa_temp_skew_method == "st-dirichlet":
                spatemp_fed_data['without_class_completion'] = \
                    {client_names[i]: st_var_cli_list[i] for i in range(len(client_names))}

        elif feat_skew_method == "hist-dirichlet":
            num_missing_classes = []
            # Set list to append labels and features for each client
            shards_no_completion = []
            shards_with_completion = []

            random_state_loop = self.random_state

            # List to append the position of the recordings extracted
            ids_list_no_completion = []
            ids_list_with_completion = []

            shape_x = np.array(np.array(image_list).shape)

            # Define feature selected
            if idx_feat == 'feat-mean':
                feature_selected = np.mean(np.array(image_list).reshape((shape_x[0], np.prod(shape_x[1:]))), axis=1)
            else:
                feature_selected = np.array(image_list).reshape((shape_x[0], np.prod(shape_x[1:])))[:, idx_feat]
            # Get ventiles from feature selected
            feature_selected = pd.DataFrame(feature_selected, columns=['feature_selected'])
            feature_selected = pd.qcut(feature_selected['feature_selected'], feat_quantile, labels=False,
                                       duplicates='drop')

            feat_distro_clients_pctg, feat_distro_clients_num, feat_distro_clients_idx, num_per_node = \
                self.dirichlet_method(labels=feature_selected, local_nodes=num_clients, alpha=alpha_feat_split,
                                      random_state=random_state_loop)

            data_df = pd.DataFrame(data)
            data_df.columns = [*data_df.columns[:-1], 'class']

            fed_data = {}
            ids_list_fed_data = {}
            pctg_distr_no_completion = []
            pctg_distr_with_completion = []
            feat_pctg_distr_with_completion = []

            for i in range(num_clients):

                X = data_df.iloc[feat_distro_clients_idx[i], 0].values
                y = data_df.iloc[feat_distro_clients_idx[i], 1].values

                if isinstance(X[0], list):
                    X = np.array(X.tolist())

                X = X.tolist()
                y = y.tolist()

                # Get the index (name) of the recordings sampled
                ids_list_no_completion.append(feat_distro_clients_idx[i])

                shards_no_completion.append(list(zip(X, y)))

                # Get distribution of labels
                df_aux = pd.DataFrame(y, columns=['label']).label.value_counts().reset_index()
                df_aux.columns = ['index', 'label']
                df_node = pd.DataFrame(np.unique(label_list), columns=['index'])
                df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
                df_node['perc'] = df_node.label / sum(df_node.label)

                pctg_distr_no_completion.append(list(df_node.perc))

                # Add missing classes when sampling (mainly for extreme case percent iid = 100)
                diff_classes = list(set(label_list) - set(y))
                num_diff_classes = len(diff_classes)
                num_missing_classes.append(num_diff_classes)

                if num_diff_classes > 0:
                    for k in diff_classes:
                        vals = [idx for idx, y in enumerate(label_list) if y == k][0]

                        feat_distro_clients_idx[i] = feat_distro_clients_idx[i] + [vals]

                X = data_df.iloc[feat_distro_clients_idx[i], 0].values
                y = data_df.iloc[feat_distro_clients_idx[i], 1].values

                if isinstance(X[0], list):
                    X = np.array(X.tolist())

                X = X.tolist()
                y = y.tolist()

                # Get distribution of labels
                df_aux = pd.DataFrame(y, columns=['label']).label.value_counts().reset_index()
                df_aux.columns = ['index', 'label']
                df_node = pd.DataFrame(np.unique(label_list), columns=['index'])
                df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
                df_node['perc'] = df_node.label / sum(df_node.label)

                pctg_distr_with_completion.append(list(df_node.perc))

                # Get distribution of feature
                df_aux = pd.DataFrame(feature_selected.values[feat_distro_clients_idx[i]],
                                      columns=['feature']).feature.value_counts().reset_index()
                df_aux.columns = ['index', 'feature']
                df_node = pd.DataFrame(np.unique(feature_selected), columns=['index'])
                df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
                df_node['perc'] = df_node.feature / sum(df_node.feature)

                feat_pctg_distr_with_completion.append(list(df_node.perc))

                # Get the index (name) of the recordings sampled
                ids_list_with_completion.append(feat_distro_clients_idx[i])

                shards_with_completion.append(list(zip(X, y)))
                if self.random_state is not None:
                    random_state_loop += self.random_state + 100

            # Add elements to dictionary of federated data
            fed_data['with_class_completion'] = \
                {client_names[i]: shards_with_completion[i] for i in range(len(client_names))}
            fed_data['without_class_completion'] = \
                {client_names[i]: shards_no_completion[i] for i in range(len(client_names))}

            # Add elements to dictionary of ids list of federated data
            ids_list_fed_data['with_class_completion'] = ids_list_with_completion
            ids_list_fed_data['without_class_completion'] = ids_list_no_completion

            # Calculate Jensen-Shannon distance for labels (no completion)
            JS_dist = jensen_shannon_distance(pctg_distr_no_completion)
            # Calculate Hellinger distance for labels (no completion)
            HD_dist = hellinger_distance(pctg_distr_no_completion)
            # Calculate Earth Mover’s distance for labels (no completion)
            emd_dist = earth_movers_distance(pctg_distr_no_completion)
            distances = {'without_class_completion': {'jensen-shannon': JS_dist, 'hellinger': HD_dist,
                                                      'earth-movers': emd_dist}}

            # Calculate Jensen-Shannon distance for labels (with class completion)
            JS_dist = jensen_shannon_distance(pctg_distr_with_completion)
            # Calculate Hellinger distance for labels (with class completion)
            HD_dist = hellinger_distance(pctg_distr_with_completion)
            # Calculate Earth Mover’s distance for labels (with class completion)
            emd_dist = earth_movers_distance(pctg_distr_with_completion)
            distances['with_class_completion'] = {'jensen-shannon': JS_dist, 'hellinger': HD_dist,
                                                  'earth-movers': emd_dist}

            # Calculate Jensen-Shannon distance for features (no completion)
            JS_dist_feat = jensen_shannon_distance(feat_distro_clients_pctg)
            # Calculate Hellinger distance for features (no completion)
            HD_dist_feat = hellinger_distance(feat_distro_clients_pctg)

            # Calculate Earth Mover’s distance for features (no completion)
            emd_dist_feat = earth_movers_distance(feat_distro_clients_pctg)

            distances['without_class_completion_feat'] = {'jensen-shannon': JS_dist_feat, 'hellinger': HD_dist_feat,
                                                          'earth-movers': emd_dist_feat}
            # Calculate Jensen-Shannon distance for features (with class completion)
            JS_dist_feat = jensen_shannon_distance(feat_pctg_distr_with_completion)
            # Calculate Hellinger distance for features (with class completion)
            HD_dist_feat = hellinger_distance(feat_pctg_distr_with_completion)
            # Calculate Earth Mover’s distance for features (with class completion)
            emd_dist_feat = earth_movers_distance(feat_pctg_distr_with_completion)

            distances['with_class_completion_feat'] = {'jensen-shannon': JS_dist_feat, 'hellinger': HD_dist_feat,
                                                       'earth-movers': emd_dist_feat}
        else:
            raise ValueError("Method '" + feat_skew_method +
                             "' not implemented. Available feature skew methods are: ['gaussian-noise', "
                             "'hist-dirichlet']")

        # Get sizes of each client
        sizes = [len(value) for key, value in fed_data['without_class_completion'].items()]
        perc_part_cli = [[elm / sum(sizes)] for elm in sizes]

        # Calculate Jensen-Shannon distance for quantity (no completion)
        JS_dist_quant = jensen_shannon_distance(perc_part_cli)
        # Calculate Hellinger distance for quantity (no completion)
        HD_dist_quant = hellinger_distance(perc_part_cli)
        # Calculate Earth Mover’s distance for quantity (no completion)
        emd_dist_quant = earth_movers_distance(perc_part_cli)

        distances['without_class_completion_quant'] = {'jensen-shannon': JS_dist_quant, 'hellinger': HD_dist_quant,
                                                       'earth-movers': emd_dist_quant}

        # Get sizes of each client
        sizes = [len(value) for key, value in fed_data['with_class_completion'].items()]
        perc_part_cli = [[elm / sum(sizes)] for elm in sizes]

        # Calculate Jensen-Shannon distance for quantity (with class completion)
        JS_dist_quant = jensen_shannon_distance(perc_part_cli)
        # Calculate Hellinger distance for quantity (with class completion)
        HD_dist_quant = hellinger_distance(perc_part_cli)
        # Calculate Earth Mover’s distance for quantity (with class completion)
        emd_dist_quant = earth_movers_distance(perc_part_cli)

        distances['with_class_completion_quant'] = {'jensen-shannon': JS_dist_quant, 'hellinger': HD_dist_quant,
                                                    'earth-movers': emd_dist_quant}

        if spa_temp_skew_method == "st-dirichlet":
            # Spatio Temporal Skew distances
            # Calculate Jensen-Shannon distance
            JS_dist_spatemp = jensen_shannon_distance(st_var_dist_cli_pctg)
            # Calculate Hellinger distance
            H_dist_spatemp = hellinger_distance(st_var_dist_cli_pctg)
            # Calculate Earth Mover’s distance
            emd_dist_spatemp = earth_movers_distance(st_var_dist_cli_pctg)

            distances['without_class_completion_spatemp'] = {'jensen-shannon': JS_dist_spatemp,
                                                          'hellinger': H_dist_spatemp,
                                                          'earth-movers': emd_dist_spatemp}
            return fed_data, ids_list_fed_data, num_missing_classes, distances, spatemp_fed_data

        else:
            return fed_data, ids_list_fed_data, num_missing_classes, distances
