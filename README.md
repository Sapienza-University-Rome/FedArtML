# FedArtML
Federated Learning for Artificial Intelligence and Machine Learning (FedArtML) is a Python-based software library publicly available on Pypi. The library aims to facilitate Federated Learning (FL) research and simplify the comparison between centralized Machine Learning and FL research results since it allows centralized datasets' partition in a systematic and controlled way regarding label, feature and quantity skewness. In addition, the library includes existing techniques for generating federated datasets in the relevant state-of-the-art and some other proposed by the authors. Moreover, it contains various metrics for quantifying the degree of non-IID (non-IID-ness) data residing across entities participating in decentralized data.

In this repository, you can find the library's source code, the installation command, some getting-started examples (including Jupyter Notebooks), and documentation regarding its use.

Enjoy it!


# Installation
```
pip install fedartml
```

# Get started

The following are examples to start using FedArtML to partition centralized data into Federated one considering label, feature and quantity skewness. You can also check broader guides to use this tool on the [examples](https://github.com/Sapienza-University-Rome/FedArtML/tree/master/examples) folder.

## Label skew
Plotting an interactive stacked bar plot (with sliders) per each local node (client) and label's classes using the Dirichlet method.

```Python
from fedartml import InteractivePlots
from keras.datasets import cifar10

# Load CIFAR 10data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define (centralized) labels to use 
CIFAR10_labels = y_train

# Instanciate InteractivePlots object
my_plot = InteractivePlots(labels = CIFAR10_labels)

# Show plot
my_plot.show_stacked_distr_dirichlet()
```
Creating federated data from centralized data using the Dirichlet method.

```Python
from fedartml import SplitAsFederatedData
from keras.datasets import cifar10
import numpy as np

# Define random state for reproducibility
random_state = 0

# Load data
(x_train_glob, y_train_glob), (x_test_glob, y_test_glob) = cifar10.load_data()
y_train_glob = np.reshape(y_train_glob, (y_train_glob.shape[0],))
y_test_glob = np.reshape(y_test_glob, (y_test_glob.shape[0],))

# Normalize pixel values to be between 0 and 1
x_train_glob, x_test_glob = x_train_glob / 255.0, x_test_glob / 255.0

# Instantiate a SplitAsFederatedData object
my_federater = SplitAsFederatedData(random_state = random_state)

# Get federated dataset from centralized dataset
clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = my_federater.create_clients(image_list = x_train_glob, label_list = y_train_glob, 
                                                             num_clients = 2, prefix_cli='Local_node', method = "dirichlet", alpha = 1)
```

## Feature skew
Creating federated data from centralized data using the Hist-Dirichlet-based method.

```Python
from fedartml import SplitAsFederatedData
from keras.datasets import cifar10
import numpy as np

# Define random state for reproducibility
random_state = 0

# Load data
(x_train_glob, y_train_glob), (x_test_glob, y_test_glob) = cifar10.load_data()
y_train_glob = np.reshape(y_train_glob, (y_train_glob.shape[0],))
y_test_glob = np.reshape(y_test_glob, (y_test_glob.shape[0],))

# Normalize pixel values to be between 0 and 1
x_train_glob, x_test_glob = x_train_glob / 255.0, x_test_glob / 255.0

# Instantiate a SplitAsFederatedData object
my_federater = SplitAsFederatedData(random_state = random_state)

# Get federated dataset from centralized dataset
clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = my_federater.create_clients(image_list = x_train_glob, label_list = y_train_glob, 
                                                             num_clients = 2, prefix_cli='Local_node', method="no-label-skew", feat_skew_method="hist-dirichlet", alpha_feat_split = 1)
```

## Quantity skew
Creating federated data from centralized data using the MinSize-Dirichlet method.

```Python
from fedartml import SplitAsFederatedData
from keras.datasets import cifar10
import numpy as np

# Define random state for reproducibility
random_state = 0

# Load data
(x_train_glob, y_train_glob), (x_test_glob, y_test_glob) = cifar10.load_data()
y_train_glob = np.reshape(y_train_glob, (y_train_glob.shape[0],))
y_test_glob = np.reshape(y_test_glob, (y_test_glob.shape[0],))

# Normalize pixel values to be between 0 and 1
x_train_glob, x_test_glob = x_train_glob / 255.0, x_test_glob / 255.0

# Instantiate a SplitAsFederatedData object
my_federater = SplitAsFederatedData(random_state = random_state)

# Get federated dataset from centralized dataset
clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = my_federater.create_clients(image_list = x_train_glob, label_list = y_train_glob, 
                                                             num_clients = 2, prefix_cli='Local_node', method = "no-label-skew", quant_skew_method="minsize-dirichlet", alpha_quant_split=1)
```



# Documentation
Find the documentation of the library on:
https://fedartml.readthedocs.io/en/latest/index.html#
