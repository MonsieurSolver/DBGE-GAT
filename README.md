# Efficient Graph Embedding Using Dissimilarity: A GAT-based Approach

This is a Python framework for Dissimilarity-Based Graph Embedding (DBGE). The code is associated with the publication "Dissimilarity-Based Graph Embedding: An Efficient GAT-based Approach".

## Installation
To use the framework and facilitate its installation, the environment.yml is present in the project.

```
conda env create -f environment.yml
conda activate DBGE
```

## How to use

### Calculate the dissimilarity matrix
The first step is to use the code to calculate the dissimilarity matrix from a TUDortmund dataset.

```
usage: compute_dissimilarity.py [-h] --dataset DATASET [--ratio RATIO] [--test TEST]

Compute the dissimilarity matrix for a dataset.

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name
  --ratio RATIO      Percentage of the dataset to be used as a test set [optional] (default=0.1)
  --test TEST        Test number [optional] (default=0)


Example:

python compute_dissimilarity.py --dataset MUTAG --test 1 --ratio 0.1 

```


### Train the network and generate the embedding
The second step is to use the code to train the GAT. The code also produces embeddings on the test set data generated by the GAT on the dataset.

```
usage: GAT.py [-h] --dataset DATASET [--test TEST] [--epoch EPOCH] [--batch BATCH] [--lr LR]
              [--lambda_value LAMBDA_VALUE] [--verbose VERBOSE]

Train the network and predict embeddings.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name
  --test TEST           Test number [optional] (default=0)
  --epoch EPOCH         Epoch [optional] (default=250)
  --batch BATCH         Batch size [optional] (default=32)
  --lr LR               Learning rate [optional] (default=0.0001)
  --lambda_value LAMBDA_VALUE Lambda [optional] (default=0.85)
  --verbose VERBOSE     Verbose [optional] (default=0)


Example:

python GAT.py --dataset MUTAG --test 1 --epoch 250 --batch 8 --lr 0.0001 --lambda_value 0.9
```

### Classify embeddings

The third step is to use the three classifiers SVM, KNearestNeighbors, RandomForest to classify the embeddings, then checking their accuracy.

```
usage: classify.py [-h] --dataset DATASET [--test TEST]

Classify the embeddings.

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name
  --test TEST        Test number [optional] (default=0)


Example:

python classify.py --dataset MUTAG --test 1

```


## Cite
Please cite our paper if you use this code in your research project.

```

```
