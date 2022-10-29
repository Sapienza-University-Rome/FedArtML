# FedArtML
Federated Learning for Artificial Intelligence and Machine Learning

### Installation
```
pip install fedartml
```

### Get started
How to plot an interactive stacked bar plot (with sliders) per each local node (client) and label's classes.

```Python
from fedartml import InteractivePlots
from keras.datasets import mnist

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Define labels to use
my_labels = train_y

# Instantiate a InteractivePlots object
my_plot = InteractivePlots(labels=my_labels)

# Show stacked bar distribution plot
my_plot.show_stacked_distr()
```

### Documentation
Find the documentation of the library on:
https://fedartml.readthedocs.io/en/latest/