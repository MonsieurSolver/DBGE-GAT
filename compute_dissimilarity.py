import argparse
import pickle
import random
import torch

from torch_geometric.datasets import TUDataset
from utils.utils import create_folder_if_not_exists
from utils.metrics.dissimilarity_matrix import dissimilarity_matrix_compute


def compute_dissimilarity(dataset_name, ratio, test):
    """
    Computes the dissimilarity given a TUDataset

    Args:
        dataset_name (String): dataset name
        ratio (float): percentage of the dataset to be used as a test set
        test (int): test number
    """
    global dataset
    create_folder_if_not_exists("dataset")

    try:
        dataset = TUDataset("dataset/", name=dataset_name, use_node_attr=True)
    except:
        print("\nError downloading the dataset\n")
        return exit()

    dataset = [d for d in dataset]
    random.shuffle(dataset)

    size_test = int(len(dataset) * ratio)

    train_dataset = dataset[:-size_test]
    test_dataset = dataset[-size_test:]

    print("\nTraining set size", len(train_dataset))
    print("Validation set size", len(test_dataset))

    print(f"\nCompute dissimilarity matrix on training set ({len(train_dataset)}x{len(train_dataset)})")
    matrix_training, graphs_training = dissimilarity_matrix_compute(train_dataset, train_dataset)
    print(f"\nCompute dissimilarity matrix on validation set ({len(test_dataset)}x{len(train_dataset)})")
    matrix_validation, graphs_validation = dissimilarity_matrix_compute(test_dataset, train_dataset)

    create_folder_if_not_exists("data")
    create_folder_if_not_exists("data/" + dataset_name)

    torch.save(matrix_training, "data/" + dataset_name + "/training_" + str(test) + ".pt")
    torch.save(matrix_validation, "data/" + dataset_name + "/validation_" + str(test) + ".pt")

    with open("data/" + dataset_name + "/training_" + str(test) + "_graphs.pkl", 'wb') as file:
        pickle.dump(graphs_training, file)

    with open("data/" + dataset_name + "/validation_" + str(test) + "_graphs.pkl", 'wb') as file:
        pickle.dump(graphs_validation, file)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the dissimilarity matrix for a dataset.')

    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--ratio', type=float, help='Percentage of the dataset to be used as a test set [optional] ('
                                                    'default=0.1)', default=0.1)
    parser.add_argument('--test', type=int, help='Test number [optional] (default=0)', default=0)

    args = parser.parse_args()

    compute_dissimilarity(args.dataset, args.ratio, args.test)
