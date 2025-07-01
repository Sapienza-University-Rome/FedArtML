# 📊 12-lead Electrocardiogram (ECG) arrhythmia detection dataset

_A short and descriptive title for your dataset._

---

## 📄 Description

The dataset is derived from 12-lead ECG recordings from the (Physionet 2020 competition)[https://physionet.org/content/challenge-2020/1.0.2/] and transformed into a structured tabular format with more than 600 features per recording. The feature extraction process involved applying the Discrete Wavelet Transform (DWT) to capture spectral components, alongside using the original ECG signals and morphological features. From these, a wide range of features are computed, including entropy-based measures (e.g., Shannon entropy) and statistical descriptors such as mean, standard deviation, and percentiles. This transformation enables the raw time-series data to be used effectively in machine learning models, particularly within a federated learning setting.

---

## 📦 File Descriptions

- **all_features_all_datsets.csv**: Contains the link to the original dataset (650 features, xx, xx and xx)
- **all_datasets_federated.csv**: Contains a reduced version of the dataset (features selected with XGBoost feature importance)
- **name_labels_dic.pkl**: Contains the dictionary of labels (arrhythmiass) for label encoding.
- **README.md**: This documentation.

## 📬 Citation

@misc{physio2020ecg12leads,
  author = {Your Name or Organization},
  title = {12-lead Electrocardiogram (ECG) arrhythmia detection dataset},
  year = {2025},
  publisher = {GitHub},
  url = {https://your-url.com}
}
