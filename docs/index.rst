.. FedArtML documentation master file, created by
   sphinx-quickstart on Sat Oct 29 08:25:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FedArtML's documentation!
===========================================

Federated Learning for Artificial Intelligence and Machine Learning (FedArtML) is a Python-based software library publicly available on Pypi. The library aims to facilitate Federated Learning (FL) research and simplify the comparison between centralized Machine Learning and FL research results since it allows centralized datasets' partition in a systematic and controlled way regarding label, feature and quantity skewness. In addition, the library includes existing techniques for generating federated datasets in the relevant state-of-the-art and some other proposed by the authors. Moreover, it contains various metrics for quantifying the degree of non-IID (non-IID-ness) data residing across entities participating in decentralized data.

Access to the official `GitHub <https://github.com/Sapienza-University-Rome/FedArtML>`_ repo where you can find the library's source code, the installation command, some getting-started examples (including Jupyter Notebooks), and documentation regarding its use.


.. toctree::
   :maxdepth: 2
   :caption: Interactive plots:

   source/api/InteractivePlots

.. toctree::
   :maxdepth: 2
   :caption: Create Federated Data:

   source/api/SplitAsFederatedData