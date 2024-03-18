import os
import torch
import pickle
import numpy as np

from torch_geometric.loader import DataLoader


def create_folder_if_not_exists(folder_path):
    """
    If it does not exist, create the folder

    Args:
        folder_path (String): dataset name
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def getDataLoader(dataset_name, test, batch_size):
    """
    Loads data and generates loaders to train the network.

    Args:
        dataset_name (String): dataset name
        test (int): test number
        batch_size (int): Batch size for the training set

    Returns:
        size_attr (int): number of node attributes
        size_lat (int): size of the final embedding
        train_lat.mean() (float): training set GED-Embedding mean
        train_lat.std() (float): training set GED-Embedding standard deviation
        train_loader (DataLoader): training set loader
        val_loader (DataLoader): test set loader
        train_graph (list): list of graphs (NetworkX) of training set
        validation_graph (list): list of graphs (NetworkX) of test set
    """

    try:
        train_ged = torch.load("data/" + dataset_name + "/training_" + str(test) + ".pt")
        validation_ged = torch.load("data/" + dataset_name + "/validation_" + str(test) + ".pt")

        with open("data/" + dataset_name + "/training_" + str(test) + "_graphs.pkl", 'rb') as file:
            train_graph = pickle.load(file)

        with open("data/" + dataset_name + "/validation_" + str(test) + "_graphs.pkl", 'rb') as file:
            validation_graph = pickle.load(file)

        size_attr = train_ged[0].x.shape[-1]
        size_lat = train_ged[0].y.shape[0]

        train_lat = np.array([t.y.numpy() for t in train_ged])

        train_loader = DataLoader(train_ged, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_ged, batch_size=len(validation_ged) + 1)

        return (size_attr, size_lat, train_lat.mean(), train_lat.std(), train_loader, val_loader,
                train_graph, validation_graph)
    except:
        print("Error loading data")
        exit()


def loadEmbedding(dataset_name, test):
    """
    Loads the embeddings and returns the training/test sets and their respective labels

    Args:
        dataset_name (String): dataset name
        test (int): test number

    Returns:
        train_lat (Numpy.array): GED-Embedding of training set
        train_y_real (Numpy.array): labels associated with the GED-Embedding of training set
        validation_lat (Numpy.array): GED-Embedding of test set
        gat_ged_embedding (Numpy.array): GAT-Embedding of test set
        validation_y_real (Numpy.array): labels associated with the GED-Embedding of test set
    """
    try:
        gat_ged_embedding = np.load("embedding/" + dataset_name + "/test_" + str(test) + "_GAT_embedding.npy")

        train_data = torch.load("data/" + dataset_name + "/training_" + str(test) + ".pt")
        validation_data = torch.load("data/" + dataset_name + "/validation_" + str(test) + ".pt")

        train_lat = np.array([t.y.numpy() for t in train_data])
        validation_lat = np.array([t.y.numpy() for t in validation_data])

        train_y_real = np.array([t.y_real.item() for t in train_data]).reshape(-1)
        validation_y_real = np.array([t.y_real.item() for t in validation_data]).reshape(-1)

        X_train_mean = train_lat.mean()
        X_train_std = train_lat.std()
        train_lat = (train_lat - X_train_mean) / X_train_std
        validation_lat = (validation_lat - X_train_mean) / X_train_std

        return train_lat, train_y_real, validation_lat, gat_ged_embedding, validation_y_real
    except:
        print("Error loading data")
        exit()
