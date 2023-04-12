# from fedartml import InteractivePlots
# from keras.datasets import mnist
# # import pandas as pd
#
# # Load MNIST data
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
#
# # Define labels to use
# my_labels = train_y
# my_x = list(train_X)
#
# # Instantiate a InteractivePlots object
# my_plot = InteractivePlots(labels=my_labels, distance="Hellinger")
# # Show stacked bar distribution plot
# my_plot.show_stacked_distr_dirichlet()

##############################################################################

from fedartml import SplitAsFederatedData
from keras.datasets import mnist
# import pandas as pd

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Define labels to use
my_labels = train_y
my_x = list(train_X)

my_plot = SplitAsFederatedData()

# Get federated dataset from centralized dataset
# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
#                                                                              num_clients=2, prefix_cli='Local_node',
#                                                                              method="percent_noniid", percent_noniid=50)

clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
                                                                             num_clients=4, prefix_cli='Local_node',
                                                                             method="dirichlet", alpha=1000)

print(distances)
print(miss_class_per_node)