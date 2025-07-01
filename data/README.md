# 📊 12-lead Electrocardiogram (ECG) arrhythmia detection dataset

_A short and descriptive title for your dataset._

---

## 📄 Description

The dataset is derived from 12-lead ECG recordings from the [Physionet 2020 competition](https://physionet.org/content/challenge-2020/1.0.2/) and transformed into a structured tabular format (*tabular dataset*) with more than 600 features per recording. The feature extraction process involved applying the Discrete Wavelet Transform (DWT) to capture spectral components, alongside using the original ECG signals and morphological features. From these, a wide range of features are computed, including entropy-based measures (e.g., Shannon entropy) and statistical descriptors such as mean, standard deviation, and percentiles. This transformation enables the raw time-series data to be used effectively in machine learning models, particularly within a federated learning setting.

- A more detailed explanation of the pipeline employed can be found in the paper [Application of Federated Learning Techniques for Arrhythmia Classification Using 12-Lead ECG Signals](https://arxiv.org/pdf/2208.10993).

- The application of the dataset for federated learning using FedArtML can be found in the paper [FedArtML: A Tool to Facilitate the Generation of Non-IID Datasets in a Controlled Way to Support Federated Learning Research](https://ieeexplore.ieee.org/document/10549893/).
---

## 📦 File Descriptions


- **all_features_all_datsets.csv**: Contains the link to the original dataset (650 features, db_name, id, and label)
- **all_datasets_federated.csv**: Contains a reduced version of the dataset (top 120 predicting features selected with XGBoost feature importance, db_name, id, and label)
- **name_labels_dic.pkl**: Contains the dictionary of labels (arrhythmiass) for label encoding (if needed).
- **README.md**: This documentation.

## 📬 Citation

@misc{physio2020ecg12leads,
  author = {Your Name or Organization},
  title = {12-lead Electrocardiogram (ECG) arrhythmia detection dataset},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Sapienza-University-Rome/FedArtML/tree/master/data}
}
