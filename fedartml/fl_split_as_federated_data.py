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

    def percent_noniid_method(self, labels, local_nodes, pct_noniid=0, random_state=None):
        """
        Create a federated dataset divided per each local node (client) using the Percentage of Non-IID (pctg_noniid)
        method.

            Parameters
            ----------
            labels : array-like
                The target values (class labels in classification).
            local_nodes : array-like
                Number of local nodes (clients) used in the federated learning paradigm.
            pct_noniid : float
                Percentage (between o and 100) desired of non-IID-ness for the federated data.
            random_state : int
                Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
                output across multiple function calls.

            Raises
            ------

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

            See Also
            --------

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

    def dirichlet_method(self, labels, local_nodes, alpha=0, random_state=None):
        """
        Create a federated dataset divided per each local node (client) using the Dirichlet (dirichlet) method.

            Parameters
            ----------
            labels : array-like
                The target values (class labels in classification).
            local_nodes : array-like
                Number of local nodes (clients) used in the federated learning paradigm.
            alpha : float
                Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for
                the federated data.
            random_state : int
                Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
                output across multiple function calls.

            Raises
            ------

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

            See Also
            --------

            References
            ----------
                .. [1] (dirichlet) Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
                    https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf
            Examples
            --------
        """
        labels = np.array(labels)

        min_size = 0
        num_classes = len(np.unique(labels))
        N = labels.shape[0]
        random_state_loop = random_state
        while min_size < num_classes:
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

    def create_clients(self, image_list, label_list, num_clients=5, prefix_cli='client', method="percent_noniid",
                       alpha=1000, percent_noniid=0):
        """
        Create a federated dataset divided per each local node (client) using the desired method (percent_noniid or
        dirichlet). It works only for classification problems (labels as classes).

            Parameters
            ----------
            image_list : array-like
                List of numpy arrays (or pandas dataframe) with images (i.e. features) from the centralized data.
            label_list : array-like
                The target values (class labels in classification) from the centralized data.
            num_clients : array-like
                Number of local nodes (clients) used in the federated learning paradigm.
            prefix_cli : array-like
                The clients' name prefix, e.g., client_1, client_2, etc.
            method : array-like
                Method to create the federated data. Possible options: "percent_noniid"(default) or "dirichlet"
            alpha : array-like
                Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness for
                the federated data.
            percent_noniid : array-like
                Percentage (between o and 100) desired of non-IID-ness for the federated data.
            Raises
            ------

            Returns
            -------

            fed_data : dict
                Contains features (images) and labels for each local node (client) after federating the data. Includes "with_class_completion" and "without_class_completion" cases.
            ids_list_fed_data : array-like
                Indexes of examples (partition) taken for each local node (client).
            num_missing_classes : array-like
                Number of missing classes per each local node when creating the federated dataset
            distances : array-like
                Distances calculated while measuring heterogeneity (non-IID-ness) of the label's distribution among clients. Includes "with_class_completion" and "without_class_completion" cases.

            Note: When creating federated data and setting heterogeneous distributions (i.e. high values of percent_noniid or small values of alpha), it is more likely the clients hold examples from only one class.
            Then, two cases are returned as output for fed_data and distances:
                - "with_class_completion": In this case, the clients are completed with one (random) example of each missing class for each client to have all the label's classes.
                - "without_class_completion": In this case, the clients are NOT completed with one (random) example of each missing class. Consequently, summing the number of examples of each client results in the same number of total examples (number of rows in image_list).

            See Also
            --------

            References
            ----------
                .. [1] (dirichlet) Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning0
                   https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf
                .. [2] (percent_noniid) Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398).PMLR.
                   https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf
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
        else:
            raise ValueError("Method '" + method + "' not implemented. Available methods are: ['percent_noniid', "
                                                   "'dirichlet']")
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

        fed_data = {}
        ids_list_fed_data = {}
        pctg_distr = []

        for i in range(num_clients):

            X = data_df.iloc[lbl_distro_clients_idx[i], 0].values.tolist()
            y = data_df.iloc[lbl_distro_clients_idx[i], 1].values.tolist()

            # Get the index (name) of the recordings sampled
            ids_list_no_completion.append(lbl_distro_clients_idx[i])

            shards_no_completion.append(list(zip(X, y)))

            # Add missing classes when sampling (mainly for extre case percent iid = 100)
            diff_classes = list(set(label_list) - set(lbl_distro_clients_num[i]))
            num_diff_classes = len(diff_classes)
            num_missing_classes.append(num_diff_classes)

            if num_diff_classes > 0:
                for k in diff_classes:
                    vals = [idx for idx, y in enumerate(label_list) if y == k][0]

                    lbl_distro_clients_idx[i] = lbl_distro_clients_idx[i] + [vals]

            X = data_df.iloc[lbl_distro_clients_idx[i], 0].values.tolist()
            y = data_df.iloc[lbl_distro_clients_idx[i], 1].values.tolist()

            # Get distribution of labels
            df_aux = pd.DataFrame(y, columns=['label']).label.value_counts().reset_index()
            df_node = pd.DataFrame(np.unique(y), columns=['index'])
            df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
            df_node['perc'] = df_node.label / sum(df_node.label)

            pctg_distr.append(list(df_node.perc))

            # Get the index (name) of the recordings sampled
            ids_list_with_completion.append(lbl_distro_clients_idx[i])

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

        # Calculate Jensen-Shannon distance
        JS_dist = jensen_shannon_distance(pctg_distr)
        # Calculate Hellinger distance
        H_dist = hellinger_distance(pctg_distr)
        # Calculate Earth Mover’s distance
        emd_dist = earth_movers_distance(pctg_distr)

        distances['with_class_completion'] = {'jensen-shannon': JS_dist, 'hellinger': H_dist,
                                              'earth-movers': emd_dist}

        return fed_data, ids_list_fed_data, num_missing_classes, distances
