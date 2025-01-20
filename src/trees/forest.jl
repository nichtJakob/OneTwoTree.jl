# A forest is a collection of trees wich aggregate their decisions
# First I cover forests for classification and after that forests for regression

abstract type AbstractForest end

mutable struct ForestClassifier <: AbstractForest
    trees::Vector{DecisionTreeClassifier}
    n_trees::Int #TODO: do we need to store this?
    n_features_per_tree::Int
    max_depth::Int #TODO: do we need to store this?
end

mutable struct ForestRegressor <: AbstractForest
    trees::Vector{DecisionTreeRegressor}
    n_trees::Int
    n_features_per_tree::Int
    max_depth::Int
end

function verify_forest_args(n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    if n_trees <= 0
        error("A RandomForest needs more than 0 trees.\n (Currently n_trees == $n_trees)")
    end

    if n_features_per_tree <= 0
        error("A RandomForest needs more than 0 features per Tree.\n (Currently n_features_per_tree == $n_features_per_tree)")
    end

    if max_depth <= 0
        error("A RandomForest needs more than 0 max depth per Tree.\n (Currently max_depth == $max_depth)")
    end
end

#TODO: n_features_per_tree could be defaulted to a percentage of the incoming data in fit!()
function ForestClassifier(;n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    verify_forest_args(n_trees, n_features_per_tree, max_depth)

    ForestClassifier(Vector{DecisionTreeClassifier}(), n_trees, n_features_per_tree, max_depth)
end

function ForestRegressor(;n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    verify_forest_args(n_trees, n_features_per_tree, max_depth)

    ForestRegressor(Vector{DecisionTreeRegressor}(), n_trees, n_features_per_tree, max_depth)
end

#Returns random features and their labels
function get_random_features(features::Matrix{S}, labels::Vector{T}, n_features::Int) where {S<:Union{Real, String}, T<:Union{Number, String}}
    random_indices = rand(1:size(features,1), n_features)
    random_features = features[random_indices, :]
    random_labels = labels[random_indices]
    return random_features, random_labels
end


function fit!(forest::AbstractForest, features::Matrix{S}, labels::Vector{T}, column_data=false) where {S<:Union{Real, String}, T<:Union{Number, String}}
    is_classifier = (forest isa ForestClassifier)

    for i in 1:forest.n_trees
        # get random dataset of size forest.n_features_per_tree
        current_tree_features, current_tree_labels = get_random_features(features, labels, forest.n_features_per_tree)

        if is_classifier
            tree = DecisionTreeClassifier(max_depth=forest.max_depth)
        else
            tree = DecisionTreeRegressor(max_depth=forest.max_depth)
        end

        fit!(tree, current_tree_features, current_tree_labels)
        push!(forest.trees, tree)
    end
end

function predict(forest::AbstractForest, X::Union{Matrix{S}, Vector{S}}) where S<:Union{Real, String}
    if isempty(forest.trees)
        error("Prediction failed because there are no trees. (Maybe you forgot to fit?)")
    end

    predictions = [predict(tree, X) for tree in forest.trees]

    if forest isa ForestClassifier
        return mode(predictions)
    else
        return mean(predictions)
    end
end

function _forest_to_string(forest::AbstractForest)
    result = ""
    for (i, tree) in enumerate(forest.trees)
        result *= "\nTree $i:\n"
        result *= _tree_to_string(tree, false)
        result *= "\n"
    end
    return result
end

function print_forest(forest::AbstractForest)
    print(_forest_to_string(forest))
end