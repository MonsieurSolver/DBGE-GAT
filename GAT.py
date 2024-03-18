import argparse
import numpy as np

from utils.network.model import GATModel, model_train, model_predict
from utils.utils import getDataLoader, create_folder_if_not_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the network and predict embedding.')

    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--test', type=int, help='Test number [optional] (default=0)', default=0)
    parser.add_argument('--epoch', type=int, help='Epoch [optional] (default=250)', default=250)
    parser.add_argument('--batch', type=int, help='Batch size [optional] (default=32)', default=32)
    parser.add_argument('--lr', type=float, help='Learning rate [optional] (default=0.0001)', default=0.0001)
    parser.add_argument('--lambda_value', type=float, help='Lambda [optional] (default=0.85)', default=0.85)
    parser.add_argument('--verbose', type=int, help='Verbose [optional] (default=0)', default=0)

    args = parser.parse_args()

    size_attr, size_lat, train_mean, train_std, train_loader, val_loader, train_graph, validation_graph = getDataLoader(args.dataset, args.test, args.batch)

    model_gat = GATModel(size_attr, size_lat)
    print("#################")
    print("#  GAT network  #")
    print("#################")
    print("")
    create_folder_if_not_exists("model")
    create_folder_if_not_exists("model/" + args.dataset)
    model_train(model_gat, size_lat, train_loader, args.epoch, args.lr, args.lambda_value, train_mean, train_std, args.verbose, args.dataset, args.test)

    val_ged_embedding = model_predict(model_gat, size_lat, val_loader, train_mean, train_std, args.dataset, args.test)

    create_folder_if_not_exists("embedding")
    create_folder_if_not_exists("embedding/" + args.dataset)
    np.save("embedding/"+args.dataset+"/test_"+str(args.test)+"_GAT_embedding.npy", val_ged_embedding.detach().numpy())
