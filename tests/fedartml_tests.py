import unittest
import math
from fedartml import InteractivePlots
from keras.datasets import mnist


class FedArtMLTestCase(unittest.TestCase):

    def setUp(self):
        # Load MNIST data
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        # Define labels to use
        my_labels = train_y
        self.my_plot = InteractivePlots(labels=my_labels)

    def test_show_stacked_distr(self):
        """Test stacked bar distribution plot """

        # Show plot
        self.my_plot.show_stacked_distr_dirichlet()

    def test_show_scatter_distr(self):
        """Test stacked bar distribution plot """

        # Show plot
        self.my_plot.show_scatter_distr_dirichlet()

    def test_show_bar_divided_distr(self):
        """Test stacked bar distribution plot """

        # Show plot
        self.my_plot.show_bar_divided_distr_dirichlet()


if __name__ == '__main__':
    unittest.main()
