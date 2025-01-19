```@meta
CurrentModule = OneTwoTree


# OneTwoTree

Documentation for [OneTwoTree](https://github.com/nichtJakob/OneTwoTree.jl).

```@index
```

```@autodocs
Modules = [OneTwoTree]
```
# OneTwoTree Documentation

This documentation provides an overview of the `OneTwoTree.jl` package.

## API Documentation

@tabset
```@tab Core Types and Structures
### Core Types
- `Decision`: Represents a decision with a function and parameter.
- `DecisionTreeClassifier`: A tree for classification tasks.
- `DecisionTreeRegressor`: A tree for regression tasks.
- `Node`: Represents a decision node in the tree.

```@tab Training Functions
### Training Functions
- `fit!(tree, features, labels)`: Train a decision tree using the dataset.
- `should_split(N, post_split_impurity, max_depth)`: Determines whether to split a node.
- `split(N)`: Performs the optimal split for a node.
- `split_indices(dataset, node_data, decision_fn)`: Splits indices based on a decision function.

```@tab Prediction and Evaluation
### Prediction and Evaluation
- `predict(tree, X)`: Predict outcomes for a dataset.
- `calc_accuracy(labels, predictions)`: Calculate prediction accuracy.
- `calc_depth(tree)`: Returns the depth of the tree.
- `label_mean(labels, indices)`: Compute the mean of a subset of labels.

```@tab Utility Functions
### Utility Functions
- `load_data(name)`: Load preconfigured datasets.
- `printM(M)`: Print a matrix.
- `print_tree(tree)`: Visualize the tree structure.
- `_tree_to_string(tree)`: Textual visualization of the tree.
- `_node_to_string(node, ...)`: Textual visualization of a node.

```@tab Decision Functions
### Decision Functions
- `equal`: A categorical decision function.
- `lessThanOrEqual`: A numerical decision function.