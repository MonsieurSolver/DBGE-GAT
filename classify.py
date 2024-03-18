import argparse
import warnings

from utils.utils import loadEmbedding

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


def classify(model_, param_, x_train_, y_train, x_val_, gat_val_, y_val):
    """
    Tests the classifier by measuring its classification accuracy where it has been optimized.

    Args:
        model_ (sklearn Model): model to test
        param_ (dictionary): parameters to optimize
        x_train_ (Numpy.array): GED-Embedding training set data
        y_train (Numpy.array): GED-Embedding training set label
        x_val_ (Numpy.array): GED-Embedding test set data
        gat_val_ (Numpy.array): GAT-Embedding test set data
        y_val (Numpy.array): GED-Embedding test set label
    """
    grid_search = GridSearchCV(model_, param_, cv=5, verbose=0)
    grid_search.fit(x_train_, y_train)
    print("Best parameters:", grid_search.best_params_)

    accuracy = grid_search.score(x_val_, y_val)
    print(f"Accuracy on the embedding of the validation set: {accuracy:.4f}")

    if gat_val_ is not None:
        accuracy = grid_search.score(gat_val_, y_val)
        print(f"Accuracy on the GAT embedding of the validation set: {accuracy:.4f}\n")
    else:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify the embeddings.')

    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--test', type=int, help='Test number [optional] (default=0)', default=0)

    args = parser.parse_args()

    t_lat, t_y_real, v_lat, gat_ged_embedding, v_y_real = loadEmbedding(args.dataset, args.test)

    param = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 'auto'],
        'kernel': ['rbf']
    }
    model = SVC()
    print("\nSVC GED Embedding / GAT Embedding")
    classify(model, param, t_lat, t_y_real, v_lat, gat_ged_embedding, v_y_real)

    param = {
        'n_neighbors': [3, 5, 7, 10],
    }
    model = KNeighborsClassifier()
    print("\nKNeighborsClassifier GED Embedding / GAT Embedding")
    classify(model, param, t_lat, t_y_real, v_lat, gat_ged_embedding, v_y_real)

    param = {
        'n_estimators': [50, 100],
        'criterion': ['gini'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 3, 5],
    }
    model = RandomForestClassifier()
    print("\nRandomForestClassifier GED Embedding / GAT Embedding")
    classify(model, param, t_lat, t_y_real, v_lat, gat_ged_embedding, v_y_real)
