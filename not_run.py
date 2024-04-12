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
##############################
###### CIFAR10 DATASET ######
##############################

# from fedartml import SplitAsFederatedData
# from keras.datasets import cifar10
# import numpy as np
# random_state = 0
# # Load CIFAR 10data
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = np.reshape(y_train, (y_train.shape[0],))
# y_test = np.reshape(y_test, (y_test.shape[0],))
# x_train = x_train / 255
# x_test = x_test / 255
#
# # Define (centralized) labels to use
# CIFAR10_labels = y_train
#
# # Instanciate InteractivePlots object
# my_plot = SplitAsFederatedData(random_state=random_state)
#
#
# # # Label skew dirichlet
# # alphas = [0, 0.000001, 0.00001,0.0001,0.0005,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# alphas = [1000]
# for alp in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                              num_clients=6, prefix_cli='Local_node',
#                                                                              method="dirichlet", alpha=alp)
#     del clients_glob,list_ids_sampled,miss_class_per_node,distances
# # print(distances)

#
# # # Feature skew Gaussian Noise
# sigmas = [0, 0.000001, 0.00001,0.0001,0.0005,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# sigmas = [0.02]
# for sig in sigmas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                              num_clients=6, prefix_cli='Local_node',
#                                                                              method="dirichlet", alpha=1000, sigma_noise=sig, bins='n_samples',feat_sample_rate=0.01)
#     del clients_glob,list_ids_sampled,miss_class_per_node,distances
# # print(distances)
# # print(miss_class_per_node)
#
# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=1000,
#                                                                                         feat_skew_method="feature-split",
#                                                                                         alpha_feat_split=0.01)
# print(distances)


# # Feature skew hist-dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# alphas = [1000,6,1,0.3,0.03]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=1000,
#                                                                                         feat_skew_method="hist-dirichlet",
#                                                                                         alpha_feat_split=alpha_sel)

# print(distances)
# print(distances['without_class_completion_feat']['hellinger'])

# # Quantity skew dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # # alphas = [0.01,0.001,0.0001]
# # alphas = [0.03]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         alpha=1000,
#                                                                                         quant_skew_method="dirichlet",
#                                                                                         alpha_quant_split=alpha_sel)
# # # print(distances)
# print(distances['without_class_completion_quant'])
# print(distances['without_class_completion'])

# Dirichlet Label Temporal skew (DLTS)
# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train,
#                                                                                         label_list=y_train,
#                                                                                         num_clients=4,
#                                                                                         prefix_cli='Local_node',
#                                                                                         # method="no-label-skew",
#                                                                                         method="dirichlet",
#                                                                                         alpha=0.5,
#                                                                                         # temp_skew_method="DLTS",
#                                                                                         )
# print(distances)
# ####################################################################################################################
##############################
###### PHYSIONET DATASET ######
##############################

from fedartml import SplitAsFederatedData, InteractivePlots

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from io import BytesIO
import requests
import pickle
from sklearn.model_selection import train_test_split
random_state =0
#Import curated data
df_selected_data = pd.read_csv('data/all_datasets_federated.csv', sep = ";")
# Get feature names
feature_names = [x for x in df_selected_data.columns if x not in ['db_name', 'label', 'id']]
# Get features
features = df_selected_data.loc[:, df_selected_data.columns.isin(feature_names)]
# Get labels
labels = list(df_selected_data['label'])
# Replace NaN values with mean values
imputer = SimpleImputer().fit(features)
features = imputer.transform(features)
# Define min max scaler
scaler = RobustScaler()
# # Transform data
features = scaler.fit_transform(features).tolist()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_selected_data['db_name'])
spa_temp_var_glob_num = np.array(le.transform(df_selected_data['db_name']))

# Bring back the database into the features
features = np.concatenate([np.array(features),spa_temp_var_glob_num.reshape(df_selected_data.shape[0], 1)], axis=1)

# Divide data in train and an auxiliar for validation/test randomly, taking the train size as 90% of the whole data
x_train_glob, x_test_glob, y_train_glob, y_test_glob = train_test_split(features, labels, test_size = 0.1, random_state = random_state)
# Divide auxiliar data in valdiation/train randomly, taking the validation and train size as 15% (for each) of the whole data
x_val_glob, x_test_glob, y_val_glob, y_test_glob = train_test_split(x_test_glob, y_test_glob, test_size = 0.5, random_state = random_state)

# Get spatio temporal feature
spa_temp_var_glob_num_train = pd.Series(x_train_glob[:, -1])
spa_temp_var_glob_num_val = pd.Series(x_val_glob[:, -1])
spa_temp_var_glob_num_test = pd.Series(x_test_glob[:, -1])

# Delete spatio temporal variable from features
x_train_glob = x_train_glob[:, :-1].tolist()
x_val_glob = x_val_glob[:, :-1].tolist()
x_test_glob = x_test_glob[:, :-1].tolist()

# print("X Train shape:",pd.DataFrame(x_train_glob).shape)
# print("Y Train shape:",pd.DataFrame(y_train_glob).shape)
# print("Spatio Temporal variable Train shape:",pd.DataFrame(spa_temp_var_glob_num_train).shape)
# print("X Validation shape:",pd.DataFrame(x_val_glob).shape)
# print("Y Validation shape:",pd.DataFrame(y_val_glob).shape)
# print("Spatio Temporal variable Val shape:",pd.DataFrame(spa_temp_var_glob_num_val).shape)
# print("X Test shape:",pd.DataFrame(x_test_glob).shape)
# print("Y Test shape:",pd.DataFrame(y_test_glob).shape)
# print("Spatio Temporal variable Test shape:",pd.DataFrame(spa_temp_var_glob_num_test).shape)

# Define old and new labels to make change
old_label = np.unique(labels)
num_classes = len(old_label)
new_label = list(range(num_classes))
# Create dictionary and change names (from abbreviation to number)
mLink = 'https://github.com/Sapienza-University-Rome/FedArtML/blob/9e05b2b4ddf6a0cbe5edee43126969c2e9fa9e01/data/name_labels_dic.pkl?raw=True'
mfile = BytesIO(requests.get(mLink).content)
name_labels_dic = pickle.load(mfile)
# Change categories by numbers
y_train_glob_num = pd.Series([v for d in y_train_glob for k, v in name_labels_dic.items() if d == k])
y_val_glob_num = pd.Series([v for d in y_val_glob for k, v in name_labels_dic.items() if d == k])
y_test_glob_num = pd.Series([v for d in y_test_glob for k, v in name_labels_dic.items() if d == k])
# print(pd.DataFrame(x_train_glob).describe())


my_plot = SplitAsFederatedData(random_state=random_state)
my_plot = SplitAsFederatedData()

# # Label skew dirichlet
# alphas = [0.065,0.09,0.17]
# alphas = [0.03]
# for alp in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob,
#                                                                                         label_list=y_train_glob_num,
#                                                                                         num_clients=2,
#                                                                                         prefix_cli='Local_node',
#                                                                                         method="dirichlet", alpha=alp)
#
# print(distances)

# # Feature skew Gaussian Noise
# sigmas = [0, 0.000001, 0.00001,0.0001, 0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# sigmas = [10**3.8,10**4.02,10**4.15,10**5]
# sigmas = [0, 9**-6, 10**-3, 20**-1, 10**3]
# sigmas = [10**-3]
# #
# for sig in sigmas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                              num_clients=2, prefix_cli='Local_node',
#                                                                              method="dirichlet", alpha=1000, sigma_noise=sig, bins='n_samples',feat_sample_rate=0.1)
#
# print(distances)

# Feature skew hist-dirichlet

# clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                          num_clients=2, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=1000,
#                                                                                         feat_skew_method="hist-dirichlet",
#                                                                                         alpha_feat_split=1000)
#
# print(distances)
# print(distances['without_class_completion_feat']['hellinger'])

# Quantity skew dirichlet
# # alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # alphas = [0.01,0.001,0.0001]
# # alphas = [1000,1,0.07]
# alphas = [1000]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                          num_clients=2, prefix_cli='Local_node',
#                                                                          method="no-label-skew", alpha=1000,
#                                                                          quant_skew_method="dirichlet", alpha_quant_split=alpha_sel)
#
# print(distances)
# print(distances['without_class_completion_quant'])
# print(distances['without_class_completion'])

# # Quantity skew minsize-dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # alphas = [0.01,0.001,0.0001]
# alphas = [0.03]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train_glob, label_list=y_train_glob_num,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         alpha=1000,
#                                                                                         quant_skew_method="minsize-dirichlet",
#                                                                                         alpha_quant_split=alpha_sel)
#     del clients_glob, list_ids_sampled, miss_class_per_node
# print(distances)
# print(distances['without_class_completion_quant'])

# Dirichlet Spatio-Temporal
# alphas = [0.3, 1, 6, 1000]
# alphas = [1000, 100, 6, 3, 1, 1.1, 0.7, 0.5, 0.3, 0.1, 0.09, 0.07, 0.05, 0.03, 0.006, 0.002]
alphas = [0.006]
for alpha_sel in alphas:
    clients_glob, list_ids_sampled, miss_class_per_node, distances, st_clients_glob = my_plot.create_clients(image_list=x_train_glob,
                                                                                        label_list=y_train_glob_num,
                                                                                        num_clients=2,
                                                                                        prefix_cli='Local_node',
                                                                                        method="no-label-skew",
                                                                                        # method="dirichlet",
                                                                                        # alpha=0.5,
                                                                                        spa_temp_skew_method="st-dirichlet",
                                                                                        alpha_spa_temp=alpha_sel,
                                                                                        spa_temp_var=spa_temp_var_glob_num_train
                                                                                        )
# print(distances)
# print(st_clients_glob)

######################################################################################################################
##############################
###### FMNIST DATASET ######
##############################

# from fedartml import SplitAsFederatedData
# from keras.datasets import fashion_mnist
# import numpy as np
# random_state = 0
# # Load FMNIST data
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# # y_train = np.reshape(y_train, (y_train.shape[0],))
# # y_test = np.reshape(y_test, (y_test.shape[0],))
# x_train = x_train / 255
# x_test = x_test / 255
#
# print("X Train shape:", x_train.shape)
# print("Y Train shape:", y_train.shape)
#
# # Define (centralized) labels to use
# CIFAR10_labels = y_train
#
# # Instanciate InteractivePlots object
# my_plot = SplitAsFederatedData(random_state=random_state)
#
# # alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# alphas = [0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.05]
# for alp in alphas:
#     # Label skew dirichlet
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train,
#                                                                                             label_list=y_train,
#                                                                                             num_clients=300,
#                                                                                             prefix_cli='Local_node',
#                                                                                             method="dirichlet", alpha=alp)
#     print(distances['without_class_completion']['hellinger'])

#     del clients_glob, list_ids_sampled, miss_class_per_node, distances


# # Feature skew Gaussian Noise
# sigmas = [0, 0.000001, 0.00001,0.0001,0.0005,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# # sigmas = [0.02]
# for sig in sigmas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                              num_clients=6, prefix_cli='Local_node',
#                                                                              method="dirichlet", alpha=1000, sigma_noise=sig, bins='n_samples',feat_sample_rate=0.1)
#     del clients_glob,list_ids_sampled,miss_class_per_node,distances
# print(distances)
# print(miss_class_per_node)


# # Feature skew hist-dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # alphas = [1000,6,1,0.3,0.03]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=1000,
#                                                                                         feat_skew_method="hist-dirichlet",
#                                                                                         alpha_feat_split=alpha_sel)
#     del clients_glob, list_ids_sampled, miss_class_per_node, distances

# print(distances)
# print(distances['without_class_completion_feat']['hellinger'])

# # Quantity skew dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # alphas = [0.01,0.001,0.0001]
# # alphas = [1000,1.1,0.09]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         alpha=1000,
#                                                                                         quant_skew_method="dirichlet",
#                                                                                         alpha_quant_split=alpha_sel)
#     del clients_glob, list_ids_sampled, miss_class_per_node
# # print(distances)
# print(distances['without_class_completion_quant'])
# print(distances['without_class_completion'])




######################################################################################################################
##############################
###### COVTYPE DATASET ######
##############################

# from fedartml import SplitAsFederatedData
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_covtype
# from sklearn.preprocessing import MinMaxScaler
#
# X, y = fetch_covtype(return_X_y = True)
#
# random_state = 0
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = random_state)
#
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
#
# # Scaling the data using the robustScaler
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
#
# # Instanciate InteractivePlots object
# my_plot = SplitAsFederatedData(random_state=random_state)

# # alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# alphas = [0.5]
# # alphas = [0.086,0.21,0.32]
# for alp in alphas:
#     # Label skew dirichlet
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train,
#                                                                                             label_list=y_train,
#                                                                                             num_clients=10,
#                                                                                             prefix_cli='Local_node',
#                                                                                             method="dirichlet", alpha=alp)
#
#     del clients_glob, list_ids_sampled, miss_class_per_node, distances

# # Feature skew Gaussian Noise
# # sigmas = [0, 0.000001, 0.000002, 0.00001,0.0001,0.0005,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# # sigmas = [0, 0.000001, 0.000002, 0.00001,0.0001, 0.001,0.01, 0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9,10**10,10**11,10**12,10**13,10**14,10**15]
# sigmas = [0.02]
# for sig in sigmas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                              num_clients=6, prefix_cli='Local_node',
#                                                                             method="dirichlet", alpha=1000, sigma_noise=sig, bins='n_samples',feat_sample_rate=0.1)
#     del clients_glob,list_ids_sampled,miss_class_per_node
# print(distances)
# # print(miss_class_per_node)
#
#
# Feature skew hist-dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# alphas = [0.5]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                          method="no-feature-skew", alpha=10,
#                                                                                         feat_skew_method="hist-dirichlet",
#                                                                                         alpha_feat_split=alpha_sel)
#     del clients_glob, list_ids_sampled, miss_class_per_node

# print(distances)
# print(distances['without_class_completion_feat']['hellinger'])
#
# Quantity skew dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# alphas = [0.01,0.001,0.0001]
# alphas = [0.5]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         alpha=10,
#                                                                                         quant_skew_method="dirichlet",
#                                                                                         alpha_quant_split=alpha_sel)
#     del clients_glob, list_ids_sampled, miss_class_per_node
# print(distances)
# print(distances['without_class_completion_quant'])
# print(distances['without_class_completion'])


# # Quantity skew minsize-dirichlet
# alphas = [1000,100,6,3,1,1.1,0.7,0.5,0.3,0.1,0.09,0.07,0.05,0.03]
# # alphas = [0.01,0.001,0.0001]
# alphas = [0.03]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances = my_plot.create_clients(image_list=x_train, label_list=y_train,
#                                                                          num_clients=6, prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         alpha=1000,
#                                                                                         quant_skew_method="minsize-dirichlet",
#                                                                                         alpha_quant_split=alpha_sel)
#     del clients_glob, list_ids_sampled, miss_class_per_node
# print(distances)
# print(distances['without_class_completion_quant'])
# print(distances['without_class_completion'])



######################################################################################################################
##############################
###### UNSW-NB15 DATASET ######
##############################
# from fedartml import SplitAsFederatedData
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_covtype
# from sklearn.preprocessing import RobustScaler
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
#
# random_state = 0
# df_selected_data = pd.read_csv('data/curated_CL_UNSW-NB15.csv', sep=";")
# # Get feature names
# feature_names = [x for x in df_selected_data.columns if x not in ['attack_cat','st_variable']]
# # Get spatio temporal variable name
# st_var = ['st_variable']
# # Get features
# features = df_selected_data.loc[:,df_selected_data.columns.isin(feature_names)]
# # Get labels
# labels = list(df_selected_data['attack_cat'])
# # Get spatio temporal variable
# spa_temp_var_glob = np.array(df_selected_data['st_variable'])
# # Replace NaN values with mean values
# imputer = SimpleImputer().fit(features)
# features = imputer.transform(features)
# # features.columns = feature_names
# # Define min max scaler
# scaler = RobustScaler()
# # # Transform data
# features = np.array(scaler.fit_transform(features).tolist())
# # features.columns = feature_names
# # Bring back the database into the features
# features = np.concatenate([features,spa_temp_var_glob.reshape(df_selected_data.shape[0], 1)], axis=1)
# # Divide data in train and an auxiliar for validation/test randomly, taking the train size as 90% of the whole data
# x_train_glob, x_test_glob, y_train_glob, y_test_glob = train_test_split(features, labels, test_size = 0.1, random_state = random_state)
# # Divide auxiliar data in valdiation/train randomly, taking the validation and train size as 15% (for each) of the whole data
# x_val_glob, x_test_glob, y_val_glob, y_test_glob = train_test_split(x_test_glob, y_test_glob, test_size = 0.5, random_state = random_state)
# # Get spatio temporal feature
# spa_temp_var_glob_train = pd.Series(x_train_glob[:, -1])
# spa_temp_var_glob_val = pd.Series(x_val_glob[:, -1])
# spa_temp_var_glob_test = pd.Series(x_test_glob[:, -1])
#
# # Delete spatio temporal variable from features
# x_train_glob = x_train_glob[:, :-1].tolist()
# x_val_glob = x_val_glob[:, :-1].tolist()
# x_test_glob = x_test_glob[:, :-1].tolist()
# # print("X Train shape:",pd.DataFrame(x_train_glob).shape)
# # print("Y Train shape:",pd.DataFrame(y_train_glob).shape)
# # print("Spatio Temporal variable Train shape:",pd.DataFrame(spa_temp_var_glob_train).shape)
# # print("X Validation shape:",pd.DataFrame(x_val_glob).shape)
# # print("Y Validation shape:",pd.DataFrame(y_val_glob).shape)
# # print("Spatio Temporal variable Val shape:",pd.DataFrame(spa_temp_var_glob_val).shape)
# # print("X Test shape:",pd.DataFrame(x_test_glob).shape)
# # print("Y Test shape:",pd.DataFrame(y_test_glob).shape)
# # print("Spatio Temporal variable Test shape:",pd.DataFrame(spa_temp_var_glob_test).shape)
# # Encode (as number) the labels
# le = LabelEncoder()
# le.fit(labels)
# y_train_glob_num = pd.Series(le.transform(y_train_glob))
# y_val_glob_num = pd.Series(le.transform(y_val_glob))
# y_test_glob_num = pd.Series(le.transform(y_test_glob))
# # Encode (as number) the spatio temporal variable
# le = LabelEncoder()
# le.fit(spa_temp_var_glob)
# spa_temp_var_glob_train_num = pd.Series(le.transform(spa_temp_var_glob_train))
# spa_temp_var_glob_val_num = pd.Series(le.transform(spa_temp_var_glob_val))
# spa_temp_var_glob_test_num = pd.Series(le.transform(spa_temp_var_glob_test))
#
# my_plot = SplitAsFederatedData(random_state=random_state)
#
# # Dirichlet Spatio-Temporal
# # alphas = [0.3, 1, 6, 1000]
# # alphas = [1000, 100, 6, 3, 1, 1.1, 0.7, 0.5, 0.3, 0.1, 0.09, 0.07, 0.05, 0.03, 0.006, 0.002]
# alphas = [0.5]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances, st_clients_glob = my_plot.create_clients(image_list=x_train_glob,
#                                                                                         label_list=y_train_glob_num,
#                                                                                         num_clients=10,
#                                                                                         prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         # method="dirichlet",
#                                                                                         # alpha=0.5,
#                                                                                         spa_temp_skew_method="st-dirichlet",
#                                                                                         alpha_spa_temp=alpha_sel,
#                                                                                         spa_temp_var=spa_temp_var_glob_train_num
#                                                                                         )
# print(distances)
# print(st_clients_glob)







######################################################################################################################
##############################
###### Android_Ransomeware DATASET ######
##############################
# from fedartml import SplitAsFederatedData
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_covtype
# from sklearn.preprocessing import RobustScaler
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
# random_state = 0
# export_curated = pd.read_csv('data/selected_curated_CL_Android_Ransomeware.csv', sep=";")
# # Get features
# features = export_curated.iloc[:,:-2].reset_index(drop=True).values
# # Get labels
# labels = list(export_curated['Label'])
# # Get spatio temporal variable
# spa_temp_var_glob = np.array(export_curated['st_variable'])
# # Bring back the database into the features
# features = np.concatenate([features,spa_temp_var_glob.reshape(export_curated.shape[0], 1)], axis=1)
#
# # Divide data in train and an auxiliar for validation/test randomly, taking the train size as 90% of the whole data
# x_train_glob, x_test_glob, y_train_glob, y_test_glob = train_test_split(features, labels, test_size = 0.1, random_state = random_state)
#
# # Divide auxiliar data in valdiation/train randomly, taking the validation and train size as 15% (for each) of the whole data
# x_val_glob, x_test_glob, y_val_glob, y_test_glob = train_test_split(x_test_glob, y_test_glob, test_size = 0.5, random_state = random_state)
#
# # Get spatio temporal feature
# spa_temp_var_glob_train = pd.Series(x_train_glob[:, -1])
# spa_temp_var_glob_val = pd.Series(x_val_glob[:, -1])
# spa_temp_var_glob_test = pd.Series(x_test_glob[:, -1])
#
# # Delete spatio temporal variable from features
# x_train_glob = x_train_glob[:, :-1].tolist()
# x_val_glob = x_val_glob[:, :-1].tolist()
# x_test_glob = x_test_glob[:, :-1].tolist()
#
# # print("X Train shape:",pd.DataFrame(x_train_glob).shape)
# # print("Y Train shape:",pd.DataFrame(y_train_glob).shape)
# # print("Spatio Temporal variable Train shape:",pd.DataFrame(spa_temp_var_glob_train).shape)
# # print("X Validation shape:",pd.DataFrame(x_val_glob).shape)
# # print("Y Validation shape:",pd.DataFrame(y_val_glob).shape)
# # print("Spatio Temporal variable Val shape:",pd.DataFrame(spa_temp_var_glob_val).shape)
# # print("X Test shape:",pd.DataFrame(x_test_glob).shape)
# # print("Y Test shape:",pd.DataFrame(y_test_glob).shape)
# # print("Spatio Temporal variable Test shape:",pd.DataFrame(spa_temp_var_glob_test).shape)
#
# # Encode (as number) the labels
# le = LabelEncoder()
# le.fit(labels)
# y_train_glob_num = pd.Series(le.transform(y_train_glob))
# y_val_glob_num = pd.Series(le.transform(y_val_glob))
# y_test_glob_num = pd.Series(le.transform(y_test_glob))
#
# # Encode (as number) the spatio temporal variable
# le = LabelEncoder()
# le.fit(spa_temp_var_glob)
# spa_temp_var_glob_train_num = pd.Series(le.transform(spa_temp_var_glob_train))
# spa_temp_var_glob_val_num = pd.Series(le.transform(spa_temp_var_glob_val))
# spa_temp_var_glob_test_num = pd.Series(le.transform(spa_temp_var_glob_test))
#
# # Set number of local nodes
# # num_local_nodes = local_nodes_glob
#
# #
# my_plot = SplitAsFederatedData(random_state=random_state)
# #
# # # Dirichlet Spatio-Temporal
# # # alphas = [0.3, 1, 6, 1000]
# alphas = [1000, 100, 6, 3, 1.1, 1, 0.7, 0.5, 0.3, 0.1, 0.09, 0.07, 0.05, 0.03, 0.006, 0.002]
# # alphas = [0.5]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances, st_clients_glob = my_plot.create_clients(image_list=x_train_glob,
#                                                                                         label_list=y_train_glob_num,
#                                                                                         num_clients=30,
#                                                                                         prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         # method="dirichlet",
#                                                                                         # alpha=0.5,
#                                                                                         spa_temp_skew_method="st-dirichlet",
#                                                                                         alpha_spa_temp=alpha_sel,
#                                                                                         spa_temp_var=spa_temp_var_glob_train_num
#                                                                                         )
# print(distances)





######################################################################################################################
##############################
###### NYC_Motor_Vehicle_Collisions DATASET ######
##############################
# from fedartml import SplitAsFederatedData
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_covtype
# from sklearn.preprocessing import RobustScaler
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder
# random_state = 0
# export_curated = pd.read_csv('data/selected_curated_CL_NYC_Motor_Vehicle_Collisions.csv', sep=";")
# # Get features
# features = export_curated.iloc[:,:-2].reset_index(drop=True).values
# # Get labels
# labels = list(export_curated['Label'])
# # Get spatio temporal variable
# spa_temp_var_glob = np.array(export_curated['st_variable'])
# # Bring back the database into the features
# features = np.concatenate([features,spa_temp_var_glob.reshape(export_curated.shape[0], 1)], axis=1)
#
# # Divide data in train and an auxiliar for validation/test randomly, taking the train size as 90% of the whole data
# x_train_glob, x_test_glob, y_train_glob, y_test_glob = train_test_split(features, labels, test_size = 0.1, random_state = random_state)
#
# # Divide auxiliar data in valdiation/train randomly, taking the validation and train size as 15% (for each) of the whole data
# x_val_glob, x_test_glob, y_val_glob, y_test_glob = train_test_split(x_test_glob, y_test_glob, test_size = 0.5, random_state = random_state)
#
# # Get spatio temporal feature
# spa_temp_var_glob_train = pd.Series(x_train_glob[:, -1])
# spa_temp_var_glob_val = pd.Series(x_val_glob[:, -1])
# spa_temp_var_glob_test = pd.Series(x_test_glob[:, -1])
#
# # Delete spatio temporal variable from features
# x_train_glob = x_train_glob[:, :-1].tolist()
# x_val_glob = x_val_glob[:, :-1].tolist()
# x_test_glob = x_test_glob[:, :-1].tolist()
#
# # print("X Train shape:",pd.DataFrame(x_train_glob).shape)
# # print("Y Train shape:",pd.DataFrame(y_train_glob).shape)
# # print("Spatio Temporal variable Train shape:",pd.DataFrame(spa_temp_var_glob_train).shape)
# # print("X Validation shape:",pd.DataFrame(x_val_glob).shape)
# # print("Y Validation shape:",pd.DataFrame(y_val_glob).shape)
# # print("Spatio Temporal variable Val shape:",pd.DataFrame(spa_temp_var_glob_val).shape)
# # print("X Test shape:",pd.DataFrame(x_test_glob).shape)
# # print("Y Test shape:",pd.DataFrame(y_test_glob).shape)
# # print("Spatio Temporal variable Test shape:",pd.DataFrame(spa_temp_var_glob_test).shape)
#
# # Encode (as number) the labels
# le = LabelEncoder()
# le.fit(labels)
# y_train_glob_num = pd.Series(le.transform(y_train_glob))
# y_val_glob_num = pd.Series(le.transform(y_val_glob))
# y_test_glob_num = pd.Series(le.transform(y_test_glob))
#
# # Encode (as number) the spatio temporal variable
# le = LabelEncoder()
# le.fit(spa_temp_var_glob)
# spa_temp_var_glob_train_num = pd.Series(le.transform(spa_temp_var_glob_train))
# spa_temp_var_glob_val_num = pd.Series(le.transform(spa_temp_var_glob_val))
# spa_temp_var_glob_test_num = pd.Series(le.transform(spa_temp_var_glob_test))
#
# # Set number of local nodes
# # num_local_nodes = local_nodes_glob
#
# #
# my_plot = SplitAsFederatedData(random_state=random_state)
# #
# # # Dirichlet Spatio-Temporal
# # # alphas = [0.3, 1, 6, 1000]
# # alphas = [1000, 100, 6, 3, 1.1, 1, 0.7, 0.5, 0.3, 0.1, 0.09, 0.07, 0.05, 0.03, 0.006, 0.002]
# alphas = [0.5]
# for alpha_sel in alphas:
#     clients_glob, list_ids_sampled, miss_class_per_node, distances, st_clients_glob = my_plot.create_clients(image_list=x_train_glob,
#                                                                                         label_list=y_train_glob_num,
#                                                                                         num_clients=6,
#                                                                                         prefix_cli='Local_node',
#                                                                                         method="no-label-skew",
#                                                                                         # method="dirichlet",
#                                                                                         # alpha=0.5,
#                                                                                         spa_temp_skew_method="st-dirichlet",
#                                                                                         alpha_spa_temp=alpha_sel,
#                                                                                         spa_temp_var=spa_temp_var_glob_train_num
#                                                                                         )

# Available datasets for spatio/temporal skew
# Temporal skew image: bigearthnet https://www.tensorflow.org/datasets/catalog/bigearthnet
# Temporal skew tabular: UNSW-NB15 https://www.kaggle.com/datasets/alextamboli/unsw-nb15
# Space skew image: Country211 https://github.com/openai/CLIP/blob/main/data/country211.md#the-country211-dataset
# Space skew tabular: Physionet 2020
