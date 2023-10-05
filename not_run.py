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

#######################################################################################################################

# from fedartml import SplitAsFederatedData
# from keras.datasets import mnist
# # import pandas as pd
# # from sklearn import preprocessing
#
# # Load MNIST data
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
#
# train_X = train_X / 255
# test_X = test_X / 255
#
# # Define labels to use
# my_labels = train_y
# my_x = list(train_X)
#
# my_plot = SplitAsFederatedData()
#
# # Get federated dataset from centralized dataset
# # clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
# #                                                                              num_clients=2, prefix_cli='Local_node',
# #                                                                              method="percent_noniid", percent_noniid=50)
#
# # clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
# #                                                                              num_clients=4, prefix_cli='Local_node',
# #                                                                              method="dirichlet", alpha=1000)
#
# # clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
# #                                                                              num_clients=4, prefix_cli='Local_node',
# #                                                                              method="dirichlet", alpha=1000, sigma_noise=0.0000001)
# # print(distances)
# # print(miss_class_per_node)
#
# # sigmas = [10**1,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# sigmas = [0,0.00001,0.0001, 0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,10**1,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# # sigmas = [10**1]
#
# for sig in sigmas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=my_x, label_list=my_labels,
#                                                                              num_clients=4, prefix_cli='Local_node',
#                                                                              method="dirichlet", alpha=1000, sigma_noise=sig, bins='unique')


######################################################################################################################

from fedartml import SplitAsFederatedData
from keras.datasets import cifar10
import numpy as np
random_state = 0
# Load CIFAR 10data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np.reshape(y_train, (y_train.shape[0],))
y_test = np.reshape(y_test, (y_test.shape[0],))
x_train = x_train / 255
x_test = x_test / 255

# Define (centralized) labels to use
CIFAR10_labels = y_train

# Instanciate InteractivePlots object
my_plot = SplitAsFederatedData(random_state=random_state)
# # sigmas = [10**1]
# # sigmas = [10**1,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# # sigmas = [0, 0.00001,0.0001, 0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# #
# # sigmas = [0.00001,0.003,0.1]
# # sigmas = [10**3]
# #
# # for sig in sigmas:
# #     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
# #                                                                              num_clients=6, prefix_cli='Local_node',
# #                                                                              method="dirichlet", alpha=1000, sigma_noise=sig, bins='n_samples')
# # print(distances)
# # print(miss_class_per_node)
#
# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=1000,
#                                                                                         feat_skew_method="feature-split",
#                                                                                         alpha_feat_split=0.01)
# print(distances)
alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# alphas = [1000,1.1,0.09]
for alpha_sel in alphas:
    clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
                                                                         num_clients=100, prefix_cli='Local_node',
                                                                                        method="no-label-skew",
                                                                                        alpha=1000,
                                                                                        quant_skew_method="dirichlet",
                                                                                        alpha_quant_split=alpha_sel)
# print(distances)
# print(distances['without_class_completion_quant'])

# ####################################################################################################################
# from fedartml import SplitAsFederatedData
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from sklearn.impute import SimpleImputer
# import pandas as pd
# import numpy as np
# from io import BytesIO
# import requests
# import pickle
# from sklearn.model_selection import train_test_split
# random_state =0
# #Import curated data
# df_selected_data = pd.read_csv('data/all_datasets_federated.csv', sep = ";")
# # Get feature names
# feature_names = [x for x in df_selected_data.columns if x not in ['db_name', 'label', 'id']]
# # Get features
# features = df_selected_data.loc[:, df_selected_data.columns.isin(feature_names)]
# # Get labels
# labels = list(df_selected_data['label'])
# # Replace NaN values with mean values
# imputer = SimpleImputer().fit(features)
# features = imputer.transform(features)
# # Define min max scaler
# scaler = RobustScaler()
# # # Transform data
# features = scaler.fit_transform(features).tolist()
# # Divide data in train and an auxiliar for validation/test randomly, taking the train size as 90% of the whole data
# x_train_glob, x_test_glob, y_train_glob, y_test_glob = train_test_split(features, labels, test_size = 0.1, random_state = random_state)
# # Divide auxiliar data in valdiation/train randomly, taking the validation and train size as 15% (for each) of the whole data
# x_val_glob, x_test_glob, y_val_glob, y_test_glob = train_test_split(x_test_glob, y_test_glob, test_size = 0.5, random_state = random_state)
# # print("X Train shape:",pd.DataFrame(x_train_glob).shape)
# # print("Y Train shape:",pd.DataFrame(y_train_glob).shape)
# # print("X Validation shape:",pd.DataFrame(x_val_glob).shape)
# # print("Y Validation shape:",pd.DataFrame(y_val_glob).shape)
# # print("X Test shape:",pd.DataFrame(x_test_glob).shape)
# # print("Y Test shape:",pd.DataFrame(y_test_glob).shape)
# # Define old and new labels to make change
# old_label = np.unique(labels)
# num_classes = len(old_label)
# new_label = list(range(num_classes))
# # Create dictionary and change names (from abbreviation to number)
# mLink = 'https://github.com/Sapienza-University-Rome/FedArtML/blob/9e05b2b4ddf6a0cbe5edee43126969c2e9fa9e01/data/name_labels_dic.pkl?raw=True'
# mfile = BytesIO(requests.get(mLink).content)
# name_labels_dic = pickle.load(mfile)
# # Change categories by numbers
# y_train_glob_num = pd.Series([v for d in y_train_glob for k, v in name_labels_dic.items() if d == k])
# y_val_glob_num = pd.Series([v for d in y_val_glob for k, v in name_labels_dic.items() if d == k])
# y_test_glob_num = pd.Series([v for d in y_test_glob for k, v in name_labels_dic.items() if d == k])
# # print(pd.DataFrame(x_train_glob).describe())
# my_plot = SplitAsFederatedData(random_state=random_state)
# my_plot = SplitAsFederatedData()


# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob,
#                                                                                         label_list=y_train_glob_num,
#                                                                                         num_clients=10,
#                                                                                         prefix_cli='Local_node',
#                                                                                         method="dirichlet", alpha=0.03)



# sigmas = [0, 0.000001, 0.00001,0.0001, 0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# sigmas = [10**3.8,10**4.02,10**4.15,10**5]
# sigmas = [0, 9**-6, 10**-3, 20**-1, 10**3]
# sigmas = [10**4]
# #
# for sig in sigmas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                              num_clients=2, prefix_cli='Local_node',
#                                                                              method="dirichlet", alpha=1000, sigma_noise=sig, bins='n_samples',feat_sample_rate=0.1)
#
# print(distances['without_class_completion_feat']['hellinger'])

# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                          num_clients=2, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=1000,
#                                                                                         feat_skew_method="feature-split",
#                                                                                         alpha_feat_split=1000)

# print(distances)
# print(distances['without_class_completion_feat']['hellinger'])

# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # alphas = [1000,1,0.07]
# # alphas = [0.07]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                          num_clients=100, prefix_cli='Local_node',
#                                                                          method="no-label-skew", alpha=1000,
#                                                                          quant_skew_method="dirichlet", alpha_quant_split=alpha_sel)

# print(distances)
# print(distances['without_class_completion_quant'])