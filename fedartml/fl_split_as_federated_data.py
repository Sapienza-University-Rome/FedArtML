# Importing libraries
import random
import numpy as np
import pandas as pd
from numpy.random import dirichlet
from fedartml.function_base import jensen_shannon_distance, hellinger_distance, earth_movers_distance
from sklearn.model_selection import StratifiedKFold


class SplitAsFederatedData:
    """
    Generate simulated interactive plots (with sliders) from the labels provided in a federated learning paradigm to
    exemplify identically and non-identically distributed labels across the local nodes (clients).

        Parameters
        ----------
        labels : array-like
            The target values (class labels in classification).
        random_state : int
            Controls the shuffling applied to the generation of pseudorandom numbers. Pass an int for reproducible
            output across multiple function calls.
        colors : list
            Colors list used to plot. Must have a length of 7 positions.
        **plot_kwargs : dict
            Keyword arguments used for customizing plots (inherited from matplotlib.pyplot).
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def percent_noniid_method(self, labels, local_nodes, pct_noniid=0, random_state=None):
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
        labels = np.array(labels)

        min_size = 0
        num_classes = len(np.unique(labels))
        N = labels.shape[0]
        net_dataidx_map = {}
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

            random_state_loop += 100

        return pctg_distr, num_distr, idx_distr, num_per_node

    def create_clients(self, image_list, label_list, num_clients=5, prefix_cli='client', method="percent_noniid",
                       alpha=1000, percent_noniid=0):
        ''' return: a dictionary with keys clients' names and value as
                    data shards - tuple of images and label lists.
            args:
                image_list: a list of numpy arrays of training images
                label_list:a list of labels for each image
                num_client: number of federated members (clients or local nodes)
                prefix_cli: the clients' name prefix, e.g, client_1, client_2, etc.
                method: method to federate (split) data: "percent_noniid"(default) or "dirichlet"
                Alpha: concentration parameter(greater than 0) to control the identicalness from Dirichlet method
                Percent_noniid: value (between 0 and 100) to control the identicalness from Percent of non iid method
        '''
        # create a list of client names
        client_names = ['{}_{}'.format(prefix_cli, i + 1) for i in range(num_clients)]
        num_classes = len(np.unique(label_list))

        # Zip the data as list
        data = list(zip(image_list, label_list))
        # Get size of each client
        size = len(data) // num_clients
        # Set list to append the position of the recordings extracted
        ids_list = []
        num_missing_classes = []
        # Set list to append labels and features for each client
        shards_no_completion = []
        shards_with_completion = []

        random_state_loop = self.random_state

        # List to append the position of the recordings extracted
        ids_list_no_completion = []
        ids_list_with_completion = []

        if method == "dirichlet":
            lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = self.dirichlet_method(
                labels=label_list,
                local_nodes=num_clients,
                alpha=alpha,
                random_state=random_state_loop)
        else:
            lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = self.percent_noniid_method(
                labels=label_list,
                local_nodes=num_clients,
                pct_noniid=percent_noniid,
                random_state=
                random_state_loop)

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
