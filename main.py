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
# my_plot.show_stacked_distr()

# Instantiate a SplitAsFederatedData object
my_plot = SplitAsFederatedData(random_state = 0)

# print(type(train_X))

# Get federated dataset from centralized dataset
clients_glob, list_ids_sampled, miss_class_per_node = my_plot.create_clients(image_list = my_x, label_list = my_labels, num_clients = 2, initial='Local_node',
                                                        oversampled_data=False, generate_iid = False,
                                                        method = "percent_noniid", Percent_noniid = 50)

print(type(clients_glob))