from fedartml import InteractivePlots, SplitAsFederatedData
from keras.datasets import mnist
import pandas as pd

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Define labels to use
my_labels = train_y
my_x = list(train_X)
# print(my_x)
# Instantiate a InteractivePlots object
# my_plot = InteractivePlots(labels=my_labels)

# Show stacked bar distribution plot
# my_plot.show_stacked_distr_dirichlet()
# my_plot.show_scatter_distr_dirichlet()
# # Instantiate a SplitAsFederatedData object
my_plot = SplitAsFederatedData(random_state=0)

# print(type(train_X))

# Get federated dataset from centralized dataset
# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
#                                                                              num_clients=2, prefix_cli='Local_node',
#                                                                              method="percent_noniid", percent_noniid=100)

clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
                                                                             num_clients=4, prefix_cli='Local_node',
                                                                             method="dirichlet", alpha=0.01)

print(distances)
print(miss_class_per_node)