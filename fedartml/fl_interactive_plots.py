# Importing libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ipywidgets import interact, Layout, IntSlider, FloatLogSlider, FloatSlider

from fedartml.function_base import jensen_shannon_distance, hellinger_distance, earth_movers_distance, get_spaced_colors
from fedartml.fl_split_as_federated_data import SplitAsFederatedData
from sklearn import preprocessing


class InteractivePlots:
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
    distance : str
        Distance to use for measuring heterogeneity (non-IID-ness) of the label's distribution among clients.
        Possible choices: "jensen-shannon", "hellinger", "earth_movers".
    **plot_kwargs : dict
        Keyword arguments used for customizing plots (inherited from matplotlib.pyplot).
    """

    def __init__(self, labels, random_state=None, colors=None, distance="jensen-shannon", **plot_kwargs):
        if colors is None:
            colors = ["#00cfcc", "#e6013b", "#007f88", "#00cccd", "#69e0da", "darkblue", "#FFFFFF"]
        self.labels = labels
        self.n_classes = len(np.unique(labels))
        self.random_state = random_state
        self.colors = colors
        self.distance = distance
        self.plot_kwargs = plot_kwargs

    def stacked_distr_dirichlet(self, Alpha, Local_Nodes):
        """
        Create an interactive stacked bar plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        Alpha : slider
            Concentration parameter of the Dirichlet distribution.
        Local_Nodes : slider
            Number of local nodes (clients) used in the federated learning paradigm.

        Returns
        -------
        The return keyword is empty. The function shows the plot as output.
        """
        labels_encoded = self.labels

        if isinstance(self.labels[0], str):
            le = preprocessing.LabelEncoder()
            le = le.fit(self.labels)
            labels_encoded = le.transform(self.labels)

        # Get random Dirichlet distribution
        lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
            SplitAsFederatedData.dirichlet_method(labels_encoded, Local_Nodes, alpha=Alpha,
                                                  random_state=self.random_state)
        # Calculate desired distance
        if self.distance == "jensen-shannon":
            dist_select = jensen_shannon_distance(lbl_distro_clients_pctg)
            text_dist = "Jensen-Shannon"
        elif self.distance == "hellinger":
            dist_select = hellinger_distance(lbl_distro_clients_pctg)
            text_dist = "Hellinger"
        elif self.distance == "earth_movers":
            dist_select = earth_movers_distance(lbl_distro_clients_pctg)
            text_dist = "Earth Mover’s"
        else:
            raise ValueError("Distance '" + self.distance + "' not implemented. Available distances are: ["
                                                            "'jensen-shannon', 'hellinger', 'earth_movers']")

        # Defne dataframe to plot
        df_simul = pd.DataFrame(lbl_distro_clients_pctg).reset_index()
        df_simul = df_simul * 100
        df_simul['index'] = (df_simul['index'] / 100 + 1).astype(int)
        df_simul.columns = ['Local Node'] + list(np.unique(self.labels))

        # Plot
        df_simul.plot(x='Local Node', kind='bar', stacked=True,
                      **self.plot_kwargs.get('stack_plot_kwargs',
                                             {'color': get_spaced_colors(self.n_classes), 'figsize': (15, 7),
                                              'fontsize': 20, 'rot': 0, 'ylim': (0, 110)}))
        plt.legend(**self.plot_kwargs.get('stack_legend_kwargs',
                                          {'title': 'Classes', 'title_fontsize': 14, 'loc': 'center left',
                                           'bbox_to_anchor': (1.0, 0.5), 'fontsize': 12}))
        plt.xlabel(**self.plot_kwargs.get('stack_xlabel_kwargs', {'xlabel': 'Local Node', 'fontsize': 20}))
        plt.ylabel(**self.plot_kwargs.get('stack_ylabel_kwargs', {'ylabel': 'Participation (%)', 'fontsize': 20}))
        plt.title(**self.plot_kwargs.get('stack_title_kwargs',
                                         {'label': "Label's classes distribution across local nodes", 'fontsize': 25}))
        plt.text(s=text_dist + " dist. = " + str(round(dist_select, 2)),
                 **self.plot_kwargs.get('stack_text_DIST_kwargs', {'x': -0.3, 'y': 103.5, 'fontsize': 20,
                                                                   'backgroundcolor': self.colors[2],
                                                                   'color': self.colors[6]}))
        plt.show()

        return ()

    def show_stacked_distr_dirichlet(self, **slider_kwargs):
        """
        Show an interactive stacked bar plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        **slider_kwargs: dict
            Keyword arguments used for customizing sliders (inherited from ipywidgets.interact).

        Returns
        -------
        The return keyword is empty. The function shows the sliders for Alpha and number of local nodes (clients).

        References
        ----------
        .. [1] Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
               https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf

        Examples
        --------
        >>> from fedartml import InteractivePlots
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_plot = InteractivePlots(labels = train_y)
        >>> my_plot.show_stacked_distr_dirichlet()
        """
        interact(self.stacked_distr_dirichlet,
                 Alpha=FloatLogSlider(**slider_kwargs.get('alpha_slider_kwargs',
                                                          {'min': -2, 'max': 3, 'value': 1000, 'readout_format': '.4'}),
                                      layout=Layout(
                                          **slider_kwargs.get('alpha_slider_lout_kwargs', {'width': '1000px'}))),
                 Local_Nodes=IntSlider(**slider_kwargs.get('loc_nodes_slider_kwargs', {'min': 1, 'max': 10, 'step': 1,
                                                                                       'value': 4}),
                                       layout=Layout(
                                           **slider_kwargs.get('loc_nodes_slider_lout_kwargs', {'width': '1000px'}))))
        return ()

    def scatter_distr_dirichlet(self, Alpha, Local_Nodes):
        """
        Create an interactive scatter plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        Alpha : slider
            Concentration parameter of the Dirichlet distribution.
        Local_Nodes : slider
            Number of local nodes (clients) used in the federated learning paradigm.

        Returns
        -------
        The return keyword is empty. The function shows the plot as output.
        """
        labels_encoded = self.labels

        if isinstance(self.labels[0], str):
            le = preprocessing.LabelEncoder()
            le = le.fit(self.labels)
            labels_encoded = le.transform(self.labels)

        # Get random Dirichlet distribution
        lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
            SplitAsFederatedData.dirichlet_method(labels_encoded, Local_Nodes, alpha=Alpha,
                                                  random_state=self.random_state)
        # Calculate desired distance
        if self.distance == "jensen-shannon":
            dist_select = jensen_shannon_distance(lbl_distro_clients_pctg)
            text_dist = "Jensen-Shannon"
        elif self.distance == "hellinger":
            dist_select = hellinger_distance(lbl_distro_clients_pctg)
            text_dist = "Hellinger"
        elif self.distance == "earth_movers":
            dist_select = earth_movers_distance(lbl_distro_clients_pctg)
            text_dist = "Earth Mover’s"
        else:
            raise ValueError("Distance '" + self.distance + "' not implemented. Available distances are: ["
                                                            "'jensen-shannon', 'hellinger', 'earth_movers']")

        # Defne dataframe to plot
        df_simul = pd.DataFrame(num_per_node).reset_index()
        df_simul['index'] = (df_simul['index'] + 1).astype(int)
        df_simul.columns = ['Local Node'] + list(np.unique(self.labels))
        df_simul_long = pd.melt(df_simul, id_vars='Local Node',
                                value_vars=list(df_simul.columns[df_simul.columns != 'Local Node']))
        df_simul_long.sort_values(by=['variable', 'Local Node'], ascending=False, inplace=True)

        # Plot
        df_simul_long.plot.scatter(x='Local Node', y='variable', s=df_simul_long['value'],
                                   **self.plot_kwargs.get('scatter_plot_kwargs', {'figsize': (15, 8), 'fontsize': 17,
                                                                                  'xlim': (0.5, Local_Nodes + 0.5),
                                                                                  'ylim': (-2,
                                                                                           len(np.unique(
                                                                                               self.labels)) + 1),
                                                                                  'color': self.colors[0]}))
        plt.xlabel(**self.plot_kwargs.get('scatter_xlabel_kwargs', {'xlabel': 'Local Node', 'fontsize': 20}))
        plt.ylabel(**self.plot_kwargs.get('scatter_ylabel_kwargs', {'ylabel': 'Classes', 'fontsize': 20}))
        plt.title(**self.plot_kwargs.get('scatter_title_kwargs',
                                         {'label': "Number of examples across classes and local nodes",
                                          'fontsize': 25}))
        plt.text(s=text_dist + " dist. = " + str(round(dist_select, 2)),
                 **self.plot_kwargs.get('scatter_text_DIST_kwargs',
                                        {'x': 0.6, 'y': len(np.unique(self.labels)), 'fontsize': 20,
                                         'backgroundcolor': self.colors[2], 'color': self.colors[6]}))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.show()

        return ()

    def show_scatter_distr_dirichlet(self, **slider_kwargs):
        """
        Show an interactive scatter plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        **slider_kwargs: dict
            Keyword arguments used for customizing sliders (inherited from ipywidgets.interact).

        Returns
        -------
        The return keyword is empty. The function shows the sliders for Alpha and number of local nodes (clients).

        References
        ----------
        .. [1] Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
               https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf

        Examples
        --------
        >>> from fedartml import InteractivePlots
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_plot = InteractivePlots(labels = train_y)
        >>> my_plot.show_scatter_distr_dirichlet()
        """
        interact(self.scatter_distr_dirichlet,
                 Alpha=FloatLogSlider(**slider_kwargs.get('alpha_slider_kwargs',
                                                          {'min': -2, 'max': 3, 'value': 1000, 'readout_format': '.4'}),
                                      layout=Layout(
                                          **slider_kwargs.get('alpha_slider_lout_kwargs', {'width': '1000px'}))),
                 Local_Nodes=IntSlider(**slider_kwargs.get('loc_nodes_slider_kwargs', {'min': 1, 'max': 10, 'step': 1,
                                                                                       'value': 4}),
                                       layout=Layout(
                                           **slider_kwargs.get('loc_nodes_slider_lout_kwargs', {'width': '1000px'}))))
        return ()

    def bar_divided_distr_dirichlet(self, Alpha, Local_Nodes):
        """
        Create an interactive bar plot (with sliders) divided per each local node (client).

        Parameters
        ----------
        Alpha : slider
            Concentration parameter of the Dirichlet distribution.
        Local_Nodes : slider
            Number of local nodes (clients) used in the federated learning paradigm.

        Returns
        -------
        The return keyword is empty. The function shows the plot as output.
        """
        labels_encoded = self.labels

        if isinstance(self.labels[0], str):
            le = preprocessing.LabelEncoder()
            le = le.fit(self.labels)
            labels_encoded = le.transform(self.labels)

        # Get random Dirichlet distribution
        lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
            SplitAsFederatedData.dirichlet_method(labels_encoded, Local_Nodes, alpha=Alpha,
                                                  random_state=self.random_state)
        # Calculate desired distance
        if self.distance == "jensen-shannon":
            dist_select = jensen_shannon_distance(lbl_distro_clients_pctg)
            text_dist = "Jensen-Shannon"
        elif self.distance == "hellinger":
            dist_select = hellinger_distance(lbl_distro_clients_pctg)
            text_dist = "Hellinger"
        elif self.distance == "earth_movers":
            dist_select = earth_movers_distance(lbl_distro_clients_pctg)
            text_dist = "Earth Mover’s"
        else:
            raise ValueError("Distance '" + self.distance + "' not implemented. Available distances are: ["
                                                            "'jensen-shannon', 'hellinger', 'earth_movers']")

        # Defne dataframe to plot
        df_simul = pd.DataFrame(num_per_node)
        df_simul = df_simul.div(df_simul.sum(axis=1), axis=0).reset_index()
        df_simul = df_simul * 100
        df_simul['index'] = (df_simul['index'] / 100 + 1).astype(int)
        df_simul.columns = ['Local Node'] + list(np.unique(self.labels))
        df_simul_long = pd.melt(df_simul, id_vars='Local Node',
                                value_vars=list(df_simul.columns[df_simul.columns != 'Local Node']))
        df_simul_long.sort_values(by=['variable', 'Local Node'], ascending=False, inplace=True)

        # Plot
        # Define dimensions for plot
        f, axs = plt.subplots(1, Local_Nodes,
                              **self.plot_kwargs.get('bar_div_subplots_kwargs', {'figsize': (70, 20)}))

        # Initialize counter
        cont = 0

        # Loop over the clients
        for i in range(1, Local_Nodes + 1):

            group = df_simul_long[df_simul_long['Local Node'] == i]

            # Plot each client bar plot
            plt.subplot(1, Local_Nodes, cont + 1)
            plt.barh(group.variable, group['value'],
                     **self.plot_kwargs.get('bar_div_plot_kwargs', {'alpha': 1, 'color': self.colors[0]}))
            plt.xlabel(**self.plot_kwargs.get('bar_div_xlabel_kwargs', {'xlabel': 'Particip. (%)', 'fontsize': 60}))
            if i == 1:
                plt.ylabel(**self.plot_kwargs.get('bar_div_ylabel_kwargs', {'ylabel': 'Classes', 'fontsize': 60}))
            plt.title(**self.plot_kwargs.get('bar_div_title_kwargs', {'label': "Local node " + str(i), 'fontsize': 60}))
            plt.xticks(**self.plot_kwargs.get('bar_div_xticks_kwargs', {'fontsize': 30}))
            plt.yticks(**self.plot_kwargs.get('bar_div_yticks_kwargs', {'fontsize': 30}))
            plt.xlim(
                **self.plot_kwargs.get('bar_div_xlim_kwargs', {'left': 0, 'right': max(df_simul_long['value']) + 1}))

            # Increase counter
            cont += 1

        f.text(**self.plot_kwargs.get('bar_div_text_DIST_kwargs',
                                      {'x': 0.5, 'y': 0.97, 'ha': 'center', 'va': 'top', 'fontsize': 60,
                                       's': text_dist + " dist. = " + str(round(dist_select, 2)),
                                       'color': self.colors[6], 'backgroundcolor': self.colors[2]}))
        plt.show()

        return ()

    def show_bar_divided_distr_dirichlet(self, **slider_kwargs):
        """
        Show an interactive bar plot (with sliders) divided per each local node (client).

        Parameters
        ----------
        **slider_kwargs: dict
            Keyword arguments used for customizing sliders (inherited from ipywidgets.interact).

        Returns
        -------
        The return keyword is empty. The function shows the sliders for Alpha and number of local nodes (clients).

        References
        ----------
        .. [1] Tao Lin∗, Lingjing Kong∗, Sebastian U. Stich, Martin Jaggi. (2020). Ensemble Distillation for Robust Model Fusion in Federated Learning
               https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Supplemental.pdf

        Examples
        --------
        >>> from fedartml import InteractivePlots
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_plot = InteractivePlots(labels = train_y)
        >>> my_plot.show_bar_divided_distr_dirichlet()
        """
        interact(self.bar_divided_distr_dirichlet,
                 Alpha=FloatLogSlider(**slider_kwargs.get('alpha_slider_kwargs',
                                                          {'min': -2, 'max': 3, 'value': 1000, 'readout_format': '.4'}),
                                      layout=Layout(
                                          **slider_kwargs.get('alpha_slider_lout_kwargs', {'width': '1000px'}))),
                 Local_Nodes=IntSlider(**slider_kwargs.get('loc_nodes_slider_kwargs', {'min': 1, 'max': 10, 'step': 1,
                                                                                       'value': 4}),
                                       layout=Layout(
                                           **slider_kwargs.get('loc_nodes_slider_lout_kwargs', {'width': '1000px'}))))
        return ()

    def stacked_distr_percent_noniid(self, Pctg_NonIID, Local_Nodes):
        """
        Create an interactive stacked bar plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        Pctg_NonIID : slider
            Percentage (between o and 100) desired of non-IID-ness for the federated data.
        Local_Nodes : slider
            Number of local nodes (clients) used in the federated learning paradigm.

        Returns
        -------
        The return keyword is empty. The function shows the plot as output.
        """
        labels_encoded = self.labels

        if isinstance(self.labels[0], str):
            le = preprocessing.LabelEncoder()
            le = le.fit(self.labels)
            labels_encoded = le.transform(self.labels)

        # Get Percentage of NonIID method
        lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
            SplitAsFederatedData.percent_noniid_method(labels_encoded, Local_Nodes, pct_noniid=Pctg_NonIID,
                                                       random_state=self.random_state)
        # Calculate desired distance
        if self.distance == "jensen-shannon":
            dist_select = jensen_shannon_distance(lbl_distro_clients_pctg)
            text_dist = "Jensen-Shannon"
        elif self.distance == "hellinger":
            dist_select = hellinger_distance(lbl_distro_clients_pctg)
            text_dist = "Hellinger"
        elif self.distance == "earth_movers":
            dist_select = earth_movers_distance(lbl_distro_clients_pctg)
            text_dist = "Earth Mover’s"
        else:
            raise ValueError("Distance '" + self.distance + "' not implemented. Available distances are: ["
                                                            "'jensen-shannon', 'hellinger', 'earth_movers']")

        # Defne dataframe to plot
        df_simul = pd.DataFrame(lbl_distro_clients_pctg).reset_index()
        df_simul = df_simul * 100
        df_simul['index'] = (df_simul['index'] / 100 + 1).astype(int)
        df_simul.columns = ['Local Node'] + list(np.unique(self.labels))

        # Plot
        df_simul.plot(x='Local Node', kind='bar', stacked=True,
                      **self.plot_kwargs.get('stack_plot_kwargs',
                                             {'color': get_spaced_colors(self.n_classes), 'figsize': (15, 7),
                                              'fontsize': 20, 'rot': 0, 'ylim': (0, 110)}))
        plt.legend(**self.plot_kwargs.get('stack_legend_kwargs',
                                          {'title': 'Classes', 'title_fontsize': 14, 'loc': 'center left',
                                           'bbox_to_anchor': (1.0, 0.5), 'fontsize': 12}))
        plt.xlabel(**self.plot_kwargs.get('stack_xlabel_kwargs', {'xlabel': 'Local Node', 'fontsize': 20}))
        plt.ylabel(**self.plot_kwargs.get('stack_ylabel_kwargs', {'ylabel': 'Participation (%)', 'fontsize': 20}))
        plt.title(**self.plot_kwargs.get('stack_title_kwargs',
                                         {'label': "Label's classes distribution across local nodes", 'fontsize': 25}))
        plt.text(s=text_dist + " dist. = " + str(round(dist_select, 2)),
                 **self.plot_kwargs.get('stack_text_DIST_kwargs', {'x': -0.3, 'y': 103.5, 'fontsize': 20,
                                                                   'backgroundcolor': self.colors[2],
                                                                   'color': self.colors[6]}))
        plt.show()

        return ()

    def show_stacked_distr_percent_noniid(self, **slider_kwargs):
        """
        Show an interactive stacked bar plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        **slider_kwargs: dict
            Keyword arguments used for customizing sliders (inherited from ipywidgets.interact).

        Returns
        -------
        The return keyword is empty. The function shows the sliders for Pctg_noniid and number of local nodes
        (clients).

        References
        ----------
        .. [1] Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398). PMLR.
               https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf

        Examples
        --------
        >>> from fedartml import InteractivePlots
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_plot = InteractivePlots(labels = train_y)
        >>> my_plot.show_stacked_distr_percent_noniid()
        """
        interact(self.stacked_distr_percent_noniid,
                 Pctg_NonIID=FloatSlider(**slider_kwargs.get('pctg_noniid_slider_kwargs',
                                                             {'min': 0, 'max': 100, 'value': 0,
                                                              'readout_format': '.4'}),
                                         layout=Layout(
                                             **slider_kwargs.get('pctg_noniid_slider_lout_kwargs',
                                                                 {'width': '1000px'}))),
                 Local_Nodes=IntSlider(**slider_kwargs.get('loc_nodes_slider_kwargs', {'min': 1, 'max': 10, 'step': 1,
                                                                                       'value': 4}),
                                       layout=Layout(
                                           **slider_kwargs.get('loc_nodes_slider_lout_kwargs', {'width': '1000px'}))))
        return ()

    def scatter_distr_percent_noniid(self, Pctg_NonIID, Local_Nodes):
        """
        Create an interactive scatter plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        Pctg_NonIID : slider
            Percentage (between o and 100) desired of non-IID-ness for the federated data.
        Local_Nodes : slider
            Number of local nodes (clients) used in the federated learning paradigm.

        Returns
        -------
        The return keyword is empty. The function shows the plot as output.
        """
        labels_encoded = self.labels

        if isinstance(self.labels[0], str):
            le = preprocessing.LabelEncoder()
            le = le.fit(self.labels)
            labels_encoded = le.transform(self.labels)

        # Get Percentage of NonIID method
        lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
            SplitAsFederatedData.percent_noniid_method(labels_encoded, Local_Nodes, pct_noniid=Pctg_NonIID,
                                                       random_state=self.random_state)
        # Calculate desired distance
        if self.distance == "jensen-shannon":
            dist_select = jensen_shannon_distance(lbl_distro_clients_pctg)
            text_dist = "Jensen-Shannon"
        elif self.distance == "hellinger":
            dist_select = hellinger_distance(lbl_distro_clients_pctg)
            text_dist = "Hellinger"
        elif self.distance == "earth_movers":
            dist_select = earth_movers_distance(lbl_distro_clients_pctg)
            text_dist = "Earth Mover’s"
        else:
            raise ValueError("Distance '" + self.distance + "' not implemented. Available distances are: ["
                                                            "'jensen-shannon', 'hellinger', 'earth_movers']")

        # Defne dataframe to plot
        df_simul = pd.DataFrame(num_per_node).reset_index()
        df_simul['index'] = (df_simul['index'] + 1).astype(int)
        df_simul.columns = ['Local Node'] + list(np.unique(self.labels))
        df_simul_long = pd.melt(df_simul, id_vars='Local Node',
                                value_vars=list(df_simul.columns[df_simul.columns != 'Local Node']))
        df_simul_long.sort_values(by=['variable', 'Local Node'], ascending=False, inplace=True)

        # Plot
        df_simul_long.plot.scatter(x='Local Node', y='variable', s=df_simul_long['value'],
                                   **self.plot_kwargs.get('scatter_plot_kwargs', {'figsize': (15, 8), 'fontsize': 17,
                                                                                  'xlim': (0.5, Local_Nodes + 0.5),
                                                                                  'ylim': (-2,
                                                                                           len(np.unique(
                                                                                               self.labels)) + 1),
                                                                                  'color': self.colors[0]}))
        plt.xlabel(**self.plot_kwargs.get('scatter_xlabel_kwargs', {'xlabel': 'Local Node', 'fontsize': 20}))
        plt.ylabel(**self.plot_kwargs.get('scatter_ylabel_kwargs', {'ylabel': 'Classes', 'fontsize': 20}))
        plt.title(**self.plot_kwargs.get('scatter_title_kwargs',
                                         {'label': "Number of examples across classes and local nodes",
                                          'fontsize': 25}))
        plt.text(s=text_dist + " dist. = " + str(round(dist_select, 2)),
                 **self.plot_kwargs.get('scatter_text_DIST_kwargs',
                                        {'x': 0.6, 'y': len(np.unique(self.labels)), 'fontsize': 20,
                                         'backgroundcolor': self.colors[2], 'color': self.colors[6]}))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

        plt.show()

        return ()

    def show_scatter_distr_percent_noniid(self, **slider_kwargs):
        """
        Show an interactive scatter plot (with sliders) per each local node (client) and label's classes.

        Parameters
        ----------
        **slider_kwargs: dict
            Keyword arguments used for customizing sliders (inherited from ipywidgets.interact).

        Returns
        -------
        The return keyword is empty. The function shows the sliders for Pctg_NonIID and number of local nodes
        (clients).

        References
        ----------
        .. [1] Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398). PMLR.
               https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf

        Examples
        --------
        >>> from fedartml import InteractivePlots
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_plot = InteractivePlots(labels = train_y)
        >>> my_plot.show_scatter_distr_percent_noniid()
        """
        interact(self.scatter_distr_percent_noniid,
                 Pctg_NonIID=FloatSlider(**slider_kwargs.get('pctg_noniid_slider_kwargs',
                                                             {'min': 0, 'max': 100, 'value': 0,
                                                              'readout_format': '.4'}),
                                         layout=Layout(
                                             **slider_kwargs.get('pctg_noniid_slider_lout_kwargs',
                                                                 {'width': '1000px'}))),
                 Local_Nodes=IntSlider(**slider_kwargs.get('loc_nodes_slider_kwargs', {'min': 1, 'max': 10, 'step': 1,
                                                                                       'value': 4}),
                                       layout=Layout(
                                           **slider_kwargs.get('loc_nodes_slider_lout_kwargs', {'width': '1000px'}))))
        return ()

    def bar_divided_distr_percent_noniid(self, Pctg_NonIID, Local_Nodes):
        """
        Create an interactive bar plot (with sliders) divided per each local node (client).

        Parameters
        ----------
        Pctg_NonIID : slider
            Percentage (between o and 100) desired of non-IID-ness for the federated data.
        Local_Nodes : slider
            Number of local nodes (clients) used in the federated learning paradigm.

        Returns
        -------
        The return keyword is empty. The function shows the plot as output.
        """
        labels_encoded = self.labels

        if isinstance(self.labels[0], str):
            le = preprocessing.LabelEncoder()
            le = le.fit(self.labels)
            labels_encoded = le.transform(self.labels)

        # Get Percentage of NonIID method
        lbl_distro_clients_pctg, lbl_distro_clients_num, lbl_distro_clients_idx, num_per_node = \
            SplitAsFederatedData.percent_noniid_method(labels_encoded, Local_Nodes, pct_noniid=Pctg_NonIID,
                                                       random_state=self.random_state)
        # Calculate desired distance
        if self.distance == "jensen-shannon":
            dist_select = jensen_shannon_distance(lbl_distro_clients_pctg)
            text_dist = "Jensen-Shannon"
        elif self.distance == "hellinger":
            dist_select = hellinger_distance(lbl_distro_clients_pctg)
            text_dist = "Hellinger"
        elif self.distance == "earth_movers":
            dist_select = earth_movers_distance(lbl_distro_clients_pctg)
            text_dist = "Earth Mover’s"
        else:
            raise ValueError("Distance '" + self.distance + "' not implemented. Available distances are: ["
                                                            "'jensen-shannon', 'hellinger', 'earth_movers']")

        # Defne dataframe to plot
        df_simul = pd.DataFrame(num_per_node)
        df_simul = df_simul.div(df_simul.sum(axis=1), axis=0).reset_index()
        df_simul = df_simul * 100
        df_simul['index'] = (df_simul['index'] / 100 + 1).astype(int)
        df_simul.columns = ['Local Node'] + list(np.unique(self.labels))
        df_simul_long = pd.melt(df_simul, id_vars='Local Node',
                                value_vars=list(df_simul.columns[df_simul.columns != 'Local Node']))
        df_simul_long.sort_values(by=['variable', 'Local Node'], ascending=False, inplace=True)

        # Plot
        # Define dimensions for plot
        f, axs = plt.subplots(1, Local_Nodes,
                              **self.plot_kwargs.get('bar_div_subplots_kwargs', {'figsize': (70, 20)}))

        # Initialize counter
        cont = 0

        # Loop over the clients
        for i in range(1, Local_Nodes + 1):

            group = df_simul_long[df_simul_long['Local Node'] == i]

            # Plot each client bar plot
            plt.subplot(1, Local_Nodes, cont + 1)
            plt.barh(group.variable, group['value'],
                     **self.plot_kwargs.get('bar_div_plot_kwargs', {'alpha': 1, 'color': self.colors[0]}))
            plt.xlabel(**self.plot_kwargs.get('bar_div_xlabel_kwargs', {'xlabel': 'Particip. (%)', 'fontsize': 60}))
            if i == 1:
                plt.ylabel(**self.plot_kwargs.get('bar_div_ylabel_kwargs', {'ylabel': 'Classes', 'fontsize': 60}))
            plt.title(**self.plot_kwargs.get('bar_div_title_kwargs', {'label': "Local node " + str(i), 'fontsize': 60}))
            plt.xticks(**self.plot_kwargs.get('bar_div_xticks_kwargs', {'fontsize': 30}))
            plt.yticks(**self.plot_kwargs.get('bar_div_yticks_kwargs', {'fontsize': 30}))
            plt.xlim(
                **self.plot_kwargs.get('bar_div_xlim_kwargs', {'left': 0, 'right': max(df_simul_long['value']) + 1}))

            # Increase counter
            cont += 1

        f.text(**self.plot_kwargs.get('bar_div_text_DIST_kwargs',
                                      {'x': 0.5, 'y': 0.97, 'ha': 'center', 'va': 'top', 'fontsize': 60,
                                       's': text_dist + " dist. = " + str(round(dist_select, 2)),
                                       'color': self.colors[6], 'backgroundcolor': self.colors[2]}))
        plt.show()

        return ()

    def show_bar_divided_distr_percent_noniid(self, **slider_kwargs):
        """
        Show an interactive bar plot (with sliders) divided per each local node (client).

        Parameters
        ----------
        **slider_kwargs: dict
            Keyword arguments used for customizing sliders (inherited from ipywidgets.interact).

        Returns
        -------
        The return keyword is empty. The function shows the sliders for Pctg_NonIID and number of local nodes
        (clients).

        References
        ----------
        .. [1] Hsieh, K., Phanishayee, A., Mutlu, O., & Gibbons, P. (2020, November). The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (pp. 4387-4398). PMLR.
               https://proceedings.mlr.press/v119/hsieh20a/hsieh20a.pdf

        Examples
        --------
        >>> from fedartml import InteractivePlots
        >>> from keras.datasets import mnist
        >>> (train_X, train_y), (test_X, test_y) = mnist.load_data()
        >>> my_plot = InteractivePlots(labels = train_y)
        >>> my_plot.show_bar_divided_distr_percent_noniid()
        """
        interact(self.bar_divided_distr_percent_noniid,
                 Pctg_NonIID=FloatSlider(**slider_kwargs.get('pctg_noniid_slider_kwargs',
                                                             {'min': 0, 'max': 100, 'value': 0,
                                                              'readout_format': '.4'}),
                                         layout=Layout(
                                             **slider_kwargs.get('pctg_noniid_slider_lout_kwargs',
                                                                 {'width': '1000px'}))),
                 Local_Nodes=IntSlider(**slider_kwargs.get('loc_nodes_slider_kwargs', {'min': 1, 'max': 10, 'step': 1,
                                                                                       'value': 4}),
                                       layout=Layout(
                                           **slider_kwargs.get('loc_nodes_slider_lout_kwargs', {'width': '1000px'}))))
        return ()
