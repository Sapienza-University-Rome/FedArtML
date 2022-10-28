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
