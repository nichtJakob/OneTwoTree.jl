# A forest is a collection of trees wich aggregate their decisions
# First I cover forests for classification and after that forests for regression

"""
Base type for classification and regression forests.

Forests are collections of trees which aggregate their decisions.
We use this abstract type to differentiate tree construction intricacies in the fit! function
"""
abstract type AbstractForest end

"""
Forest for classification problems

# Fields:
- `trees::Vector{DecisionTreeClassifier}`: A vector of decision trees.
- `n_trees::Int`: The number of trees in the forest.
- `n_features_per_tree::Int`: The number of randomly drawn features used for each tree.
- `max_depth::Int`: The upper bound for the depth of each tree.
"""
mutable struct ForestClassifier <: AbstractForest
    trees::Vector{DecisionTreeClassifier}
    n_trees::Int
    n_features_per_tree::Int
    max_depth::Int
end

"""
Forest for regression problems

# Fields:
- `trees::Vector{DecisionTreeRegressor}`: A vector of regression trees.
- `n_trees::Int`: The number of trees in the forest.
- `n_features_per_tree::Int`: The number of randomly drawn features used for each tree.
- `max_depth::Int`: The upper bound for the depth of each tree.
"""
mutable struct ForestRegressor <: AbstractForest
    trees::Vector{DecisionTreeRegressor}
    n_trees::Int
    n_features_per_tree::Int
    max_depth::Int
end

"""
    verify_forest_args(n_trees::Int, n_features_per_tree::Int, max_depth::Int)

Verifies the validity of arguments used to initialize a forest.

# Arguments:
- `n_trees::Int`: The number of trees in the forest.
- `n_features_per_tree::Int`: The number of randomly drawn features used for each tree.
- `max_depth::Int`: The upper bound for the depth of each tree.

# Errors:
Raises an error if any of the arguments are less than or equal to zero.
"""
function verify_forest_args(n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    if n_trees <= 0
        throw(ArgumentError("A RandomForest needs more than 0 trees.\n (Currently n_trees == $n_trees)"))
    end

    if n_features_per_tree <= 0
        throw(ArgumentError("A RandomForest needs more than 0 features per Tree.\n (Currently n_features_per_tree == $n_features_per_tree)"))
    end

    if max_depth <= 0
        throw(ArgumentError("A RandomForest needs more than 0 max depth per Tree.\n (Currently max_depth == $max_depth)"))
    end
end

"""
    ForestClassifier(;n_trees::Int, n_features_per_tree::Int, max_depth::Int)

Constructs a ForestClassifier instance.

# Arguments:
- `n_trees::Int`: The number of trees in the forest.
- `n_features_per_tree::Int`: The number of randomly drawn features used for each tree.
- `max_depth::Int`: The upper bound for the depth of each tree.

# Returns:
A `ForestClassifier` instance.
"""
function ForestClassifier(;n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    verify_forest_args(n_trees, n_features_per_tree, max_depth)

    ForestClassifier(Vector{DecisionTreeClassifier}(), n_trees, n_features_per_tree, max_depth)
end

"""
    ForestRegressor(;n_trees::Int, n_features_per_tree::Int, max_depth::Int)
Constructs a ForestRegressor instance.

# Arguments:
- `n_trees::Int`: The number of trees in the forest.
- `n_features_per_tree::Int`: The number of randomly drawn features used for each tree.
- `max_depth::Int`: The upper bound for the depth of each tree.

# Returns:
A `ForestRegressor` instance.
"""
function ForestRegressor(;n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    verify_forest_args(n_trees, n_features_per_tree, max_depth)

    ForestRegressor(Vector{DecisionTreeRegressor}(), n_trees, n_features_per_tree, max_depth)
end

"""
    get_random_features(features::Matrix{S}, labels::Vector{T}, n_features::Int)

Returns random features and their corresponding labels from the given dataset.

# Arguments:
- `features::Matrix{S}`: The feature matrix.
- `labels::Vector{T}`: The labels vector.
- `n_features::Int`: The number of features to draw randomly.

# Returns:
A tuple `(random_features, random_labels)` containing the randomly drawn features and labels.
"""
function get_random_features(features::Matrix{S}, labels::Vector{T}, n_features::Int) where {S<:Union{Real, String}, T<:Union{Number, String}}
    random_indices = rand(1:size(features,1), n_features)
    random_features = features[random_indices, :]
    random_labels = labels[random_indices]
    return random_features, random_labels
end


"""
    fit!(forest::AbstractForest, dataset::AbstractMatrix, labels::Vector{T}; splitting_criterion=nothing, column_data=false) where {T<:Union{Number, String}}

Trains each tree in the forest on randomly drawn subsets of test features and corresponding test labels.

# Arguments

- `forest::AbstractForest`: the forest to be trained
- `dataset::AbstractMatrix`: the training data
- `labels::Vector{Union{Number, String}}`: the target labels
- `splitting_criterion`: a function indicating some notion of gain from splitting a node. If not provided, default criteria for classification and regression are used.
- `column_data::Bool`: whether the datapoints are contained in dataset columnwise
(OneTwoTree provides the following splitting criteria for classification: gini_gain, information_gain; and for regression: variance_gain. If you'd like to define a splitting criterion yourself, you need to consider the following:

1. The function must calculate a 'gain'-value for a split of a node, meaning that larger values are considered better.
2. The function signature must conform to `my_func(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels::AbstractVector)` where parent_labels is a set of datapoint labels, which is split into two subsets true_child_labels & false_child_labels by some discriminating function. (Each label in parent_labels is contained in exactly one of the two subsets.)
"""
function fit!(forest::AbstractForest, dataset::AbstractMatrix, labels::Vector{T}; splitting_criterion=nothing, column_data=false) where {T<:Union{Number, String}}
    is_classifier = (forest isa ForestClassifier)

    for i in 1:forest.n_trees
        # get random dataset of size forest.n_features_per_tree
        current_tree_dataset, current_tree_labels = get_random_features(dataset, labels, forest.n_features_per_tree)

        if is_classifier
            tree = DecisionTreeClassifier(max_depth=forest.max_depth)
        else
            tree = DecisionTreeRegressor(max_depth=forest.max_depth)
        end

        fit!(tree, current_tree_dataset, current_tree_labels, splitting_criterion=splitting_criterion)
        push!(forest.trees, tree)
    end
end

"""
    predict(forest::AbstractForest, X::Union{Matrix{S}, Vector{S}}) where S<:Union{Real, String}

Outputs the forest-prediction for a given datapoint X.
The prediction is based on the aggregation of the tree decisions.
For agregation in a regression scenario the mean is used.
For agregation in a classification scenario the most voted class label is used.

# Arguments:
- `forest::AbstractForest`: The trained forest.
- `X::Union{Matrix{S}, Vector{S}}`: The input data for which a prediction is searched.

# Returns:
Predictions for the input data X, aggregated across all trees in the forest.

# Errors:
Raises an error if the forest contains no trained trees.
"""
function predict(forest::AbstractForest, X::Union{Matrix{S}, Vector{S}}) where S<:Union{Real, String}
    if isempty(forest.trees)
        throw(ArgumentError("Prediction failed because there are no trees. (Maybe you forgot to fit?)"))
    end

    predictions = [predict(tree, X) for tree in forest.trees]

    if forest isa ForestClassifier
        return mode(predictions)
    else
        return mean(predictions)
    end
end

"""
Converts the forest to a string.

# Arguments:
- `forest::AbstractForest`: The forest to be represented.

# Returns:
A string with numerated text representations of the trees in the forest
"""
function _forest_to_string(forest::AbstractForest)
    result = ""
    for (i, tree) in enumerate(forest.trees)
        result *= "\nTree $i:\n"
        result *= _tree_to_string(tree, false)
        result *= "\n"
    end
    return result
end

"""
Prints the numerated trees of a forest.

# Arguments:
- `forest::AbstractForest`: The forest to be printed.
"""
function print_forest(forest::AbstractForest)
    print(_forest_to_string(forest))
end

"""
Displays a string representation of the forest when shown in the REPL or other I/O streams.

# Arguments:
- `io::IO`: The I/O stream.
- `forest::AbstractForest`: The forest to be displayed.
"""
function Base.show(io::IO, forest::AbstractForest)
    print(io, _forest_to_string(forest))
end