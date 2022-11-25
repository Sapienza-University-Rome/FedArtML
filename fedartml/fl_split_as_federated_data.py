# Importing libraries
import random
import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from fedartml.function_base import jsd, get_stratified_data
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

    def percent_noniid_method(self, labels, Local_Nodes, Pct_noniid=0, random_state=None):
        # Get number of examples in the noniid part
        n_noniid = int(len(labels) * (Pct_noniid / 100))

        # sorted_labels = sorted(labels)
        sorted_labels = labels

        noniid_part_sample = list(sorted_labels[0:n_noniid])
        iid_part_sample = list(sorted_labels[n_noniid:len(labels)])

        uniq_class_noniid = np.unique(noniid_part_sample)
        n_class_per_node_noniid = len(uniq_class_noniid) // Local_Nodes

        test_distr = []
        num_distr = []
        idx_distr = []

        n_ini = 0
        n_fin = n_class_per_node_noniid

        n_total_iid = len(sorted_labels) - len(noniid_part_sample)

        # Randomly assign each example to a local node (generate random numbers from 0 to local nodes)
        np.random.seed(random_state)
        rand_lnodes_iid = np.random.randint(0, Local_Nodes, size=n_total_iid)

        # Get data for each local node
        for i in range(Local_Nodes):
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
            df_node['perc'] = df_node.label / sum(df_node.label)

            test_distr.append(list(df_node.perc))
            num_distr.append(aux_examples_node)
            idx_distr.append(idx_aux_examples_node)

            # Increase values to consider next iteration
            n_ini += n_class_per_node_noniid

            # Check if the iteration corresponds to the previous-to-last to add all the values
            if i == (Local_Nodes - 2):
                n_fin = len(uniq_class_noniid) + 1
            else:
                n_fin += n_class_per_node_noniid

        return test_distr, num_distr, idx_distr

    def create_clients(self,image_list, label_list, num_clients=5, initial='clients', oversampled_data=False,
                       generate_iid=True, Alpha=1000, method="percent_noniid", Percent_noniid=0):
        ''' return: a dictionary with keys clients' names and value as
                    data shards - tuple of images and label lists.
            args:
                image_list: a list of numpy arrays of training images
                label_list:a list of binarized labels for each image
                num_client: number of fedrated members (clients)
                initials: the clients'name prefix, e.g, clients_1

        '''
        # create a list of client names
        client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
        num_classes = len(np.unique(label_list))

        # randomize the data
        data = list(zip(image_list, label_list))
        # shard data and place at each client
        size = len(data) // num_clients
        # List to append the position of the recordings extracted
        ids_list = []
        num_missing_classes = []
        if oversampled_data == False:
            if generate_iid:
                random.seed(self.random_state)
                random.shuffle(data)

                shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

                # number of clients must equal number of shards
                assert (len(shards) == len(client_names))
            else:
                shards = []
                random_state_loop = self.random_state

                # List to append the position of the recordings extracted
                ids_list = []

                if method == "dirichlet":
                    # Get random Dirichlet distribution
                    rand_distr = np.array(
                        dirichlet.rvs([Alpha] * num_classes, size=num_clients, random_state=random_state_loop))
                else:
                    rand_distr, num_rand_distr, idx_rand_distr = self.percent_noniid_method(labels=label_list,
                                                                                            Local_Nodes=num_clients,
                                                                                            Pct_noniid=Percent_noniid,
                                                                                            random_state=
                                                                                            random_state_loop)

                # rand_distr = np.round(rand_distr, 3)

                # Calculate Jensen-Shannon distance
                JS_dist = np.sqrt(jsd(rand_distr))

                for i in range(num_clients):

                    n_per_class = rand_distr[i] * size

                    classes = np.unique(label_list)

                    data_df = pd.DataFrame(data)

                    data_df.columns = [*data_df.columns[:-1], 'class']

                    if method == "dirichlet":
                        sample = get_stratified_data(data_df, strat_var='class', strat_classes=classes,
                                                     strat_counts=n_per_class, random_state=random_state_loop)

                        # Get the index (name) of the recordings sampled
                        ids_list.append(sample.index)

                        X = sample.iloc[:, 0].values.tolist()
                        y = sample.iloc[:, 1].values.tolist()

                    else:
                        # Add missing classes when sampling (mainly for extre case percent iid = 100)
                        diff_classes = list(set(label_list) - set(num_rand_distr[i]))
                        num_diff_classes = len(diff_classes)
                        num_missing_classes.append(num_diff_classes)

                        if num_diff_classes > 0:
                            for k in diff_classes:
                                vals = [idx for idx, y in enumerate(label_list) if y == k][0]

                                idx_rand_distr[i] = idx_rand_distr[i] + [vals]

                        X = data_df.iloc[idx_rand_distr[i], 0].values.tolist()
                        y = data_df.iloc[idx_rand_distr[i], 1].values.tolist()

                        # Get the index (name) of the recordings sampled
                        ids_list.append(idx_rand_distr[i])
                    # del data_df['p']
                    random_state_loop += self.random_state + 100

                    shards.append(list(zip(X, y)))

        else:
            shards = []

            skf = StratifiedKFold(n_splits=num_clients)
            skf.get_n_splits(image_list, label_list)

            X = np.array(image_list)
            y = np.array(label_list)
            # StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
            for train_index, test_index in skf.split(X, y):
                X_train_ver, X_test_ver = X[train_index], X[test_index]
                y_train_ver, y_test_ver = y[train_index], y[test_index]
                X_test_ver = list(X_test_ver)
                y_test_ver = list(y_test_ver)

                shards.append(list(zip(X_test_ver, y_test_ver)))

        return {client_names[i]: shards[i] for i in range(len(client_names))}, ids_list, num_missing_classes