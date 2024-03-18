import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import relu, sigmoid
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool, GATConv


class GATModel(Module):
    def __init__(self, n_attr, size_vect):
        """
        Initialize the neural network

        Args:
            n_attr (int): number of node attributes
            size_vect (int): size of the final embedding
        """
        super(GATModel, self).__init__()
        self.conv1 = GATConv(n_attr, 64)
        self.conv2 = GATConv(64, 128)
        self.conv3 = GATConv(128, 256)

        self.dense1 = Linear(256 + 256, 256)
        self.dense2 = Linear(256, 128)
        self.dense3 = Linear(128, 64)
        self.dense4 = Linear(64, 64)
        self.dense5 = Linear(64, 128)
        self.dense6 = Linear(128, 256)
        self.dense7 = Linear(256, size_vect)

        self.drop = Dropout(0.1)

        self.mean = Linear(256, 64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x1 = global_max_pool(x, batch)
        x2 = global_add_pool(x, batch)
        x3 = global_mean_pool(x, batch)

        x = torch.cat((x1, x2), -1)

        x = relu(self.dense1(x))
        x = relu(self.dense2(x))
        x = relu(self.dense3(x))
        x = x * sigmoid(self.mean(x3))
        x = relu(self.dense4(x))
        x = self.drop(x)
        x = relu(self.dense5(x))
        x = relu(self.dense6(x))
        x = self.dense7(x)

        return x


def model_train(model, size_lat, train_loader, epochs, lr, lambda_val, train_mean, train_std, verbose, dataset, test):
    """
    Train the model to approximate DBGE, save the weights in model/ dataset_name /model_ test number _weights.pt

    Args:
        model (Torch model): neural network model
        size_lat (int): size of the final embedding
        train_loader (DataLoader): training set loader
        epochs (int): epoch
        lr (float): learning rate
        lambda_val (float): lambda parameter
        train_mean (float): training set mean
        train_std (float): training set standard deviation
        verbose (int): if 1 displays the training
        dataset (String): dataset name
        test (int): test number
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available")
    else:
        device = torch.device("cpu")
        print("GPU not available")

    model = model.to(device)

    # Loss
    mseLoss = nn.MSELoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    # Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    # Save best model
    best_average_mse = 1000.0

    print("Training GAT...")
    for epoch in tqdm(range(epochs)):
        model.train()

        total_MSE = 0

        for data in train_loader:
            optimizer.zero_grad()

            output = model(data.to(device))
            output = output.to(device)

            label_ged = data.y.float().view(-1, size_lat)

            label_ged_soft = torch.softmax(label_ged, -1)
            label_ged = (label_ged - train_mean) / train_std

            output_soft = F.log_softmax(output, -1)

            loss = mseLoss(output, label_ged) * lambda_val + KLDivLoss(output_soft, label_ged_soft) * (1 - lambda_val)

            total_MSE += mseLoss(output, label_ged).item()

            loss.backward()
            optimizer.step()

        average_mse = total_MSE / len(train_loader)

        if best_average_mse > average_mse:
            best_average_mse = average_mse
            torch.save(model.state_dict(), "model/" + dataset + "/model_" + str(test) + "_weights.pt")

        if verbose == 1:
            print(f'Epoch [{epoch + 1}/{epochs}],  MSE: {average_mse}')

    print(f"Best model with MSE {round(best_average_mse, 5)}\n")


def model_predict(model, size_lat, val_loader, train_mean, train_std, dataset, test):
    """

    Performs the inference by loading the weights of the saved model. Calculate the inference time and return the
    predicted embedding.

    Args:
        model (Torch model): neural network model
        size_lat (int): size of the final embedding
        val_loader (DataLoader): test set loader
        train_mean (float): training set mean
        train_std (float): training set standard deviation
        dataset (String): dataset name
        test (int): test number

    Returns:
        output (Torch.tensor): Predicted embedding
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available")
    else:
        device = torch.device("cpu")
        print("GPU not available")

    model.load_state_dict(torch.load("model/" + dataset + "/model_" + str(test) + "_weights.pt"))
    model.eval()
    model = model.to(device)

    print("Testing inference time...")
    time_list = []

    for _ in tqdm(range(1000)):
        start = time.time()
        for data in val_loader:
            model(data.to(device))
            break
        end_time = time.time() - start
        time_list.append(end_time * 1000)

    time_list = np.array(time_list)
    print(f"Avg time: {round(time_list.mean(), 4)}(+-{round(time_list.std(), 4)})")

    mseLoss = nn.MSELoss()
    total_MSE = 0
    total_RMSPE = 0
    global output

    for data in val_loader:
        output = model(data.to(device))

        label_ged = data.y.float().view(-1, size_lat)
        label_ged = (label_ged - train_mean) / train_std
        total_MSE += mseLoss(output, label_ged).item()
        total_RMSPE += (torch.sqrt(torch.mean(torch.square((label_ged - output) / (label_ged + 0.00000001))))).item()

    average_mse = total_MSE / len(val_loader)
    average_rmspe = total_RMSPE / len(val_loader)
    print(f"Validation set MSE (RMSPE): {round(average_mse, 5)} ({round(average_rmspe, 2)}%)\n")

    return output
