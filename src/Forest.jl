# A forest is a collection of trees wich aggregate their decisions
#first i cover Forests for Classification and after that Forests for regression

mutable struct ForestClassifier <: AbstractDecisionTree
    trees::Vector{AbstractDecisionTree} #TODO better DecisionTreeClassifier?
    n_trees::Int
    n_features_per_tree::Int
    max_depth::Int
end

function ForestClassifier(n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    if n_trees <= 0
        error("A RandomForest needs more than 0 trees.\n (Currently n_trees == $n_trees)")
    end

    if n_features_per_tree <= 0
        error("A RandomForest needs more than 0 features per Tree.\n (Currently n_features_per_tree == $n_features_per_tree)")
    end

    if max_depth <= 0
        error("A RandomForest needs more than 0 max depth per Tree.\n (Currently max_depth == $max_depth)")
    end

    ForestClassifier(Vector{AbstractDecisionTree}(), n_trees, n_features_per_tree, max_depth)
end

#Returns random features and their labels
function get_random_features(features::Matrix{S}, labels::Vector{T}, n_features::Int) where {S<:Union{Real, String}, T<:Union{Number, String}}
    random_indices = rand(1:size(features,1), n_features)
    random_features = features[random_indices, :]
    random_labels = labels[random_indices]
    return random_features, random_labels
end

#function fit2!(forest::ForestClassifier, features::Matrix{Union{Real, String}}, labels::Vector{Union{Real, String}})
function fit2!(forest::ForestClassifier, features::Matrix{S}, labels::Vector{T}, column_data=false) where {S<:Union{Real, String}, T<:Union{Number, String}}
    for i in 1:forest.n_trees
        println("Tree nr $i wird gefitted")
        # get random dataset of size forest.n_features_per_tree
        current_tree_features, current_tree_labels = get_random_features(features, labels, forest.n_features_per_tree)

        # TODO Regression or Classification?
        tree = DecisionTreeClassifier(max_depth=forest.max_depth)

        fit!(tree,current_tree_features, current_tree_labels)
        push!(forest.trees, tree)
    end
end


#TODO automaticly evaluate if regression or not
function predict2(forest::ForestClassifier, X::Union{Matrix{S}, Vector{S}}) where S<:Union{Real, String}
    if isempty(forest.trees)
        error("Prediction failed because there are no trees. (Maybe you forgot to fit?)")
    end
    predictions = [predict(tree, X) for tree in forest.trees]
    return mode(predictions)
end

#----------------------------------------------------------------------------------------------------
#now regression forests

mutable struct ForestRegressor <: AbstractDecisionTree
    trees::Vector{AbstractDecisionTree} #TODO better DecisionTreeRegressor?
    n_trees::Int
    n_features_per_tree::Int
    max_depth::Int
end

function ForestRegressor(n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    if n_trees <= 0
        error("A RandomForest needs more than 0 trees.\n (Currently n_trees == $n_trees)")
    end

    if n_features_per_tree <= 0
        error("A RandomForest needs more than 0 features per Tree.\n (Currently n_features_per_tree == $n_features_per_tree)")
    end

    if max_depth <= 0
        error("A RandomForest needs more than 0 max depth per Tree.\n (Currently max_depth == $max_depth)")
    end

    ForestRegressor(Vector{AbstractDecisionTree}(), n_trees, n_features_per_tree, max_depth)
end

#function fit2!(forest::ForestClassifier, features::Matrix{Union{Real, String}}, labels::Vector{Union{Real, String}})
function fit2!(forest::ForestRegressor, features::Matrix{S}, labels::Vector{T}, column_data=false) where {S<:Union{Real, String}, T<:Union{Number, String}}
    for i in 1:forest.n_trees
        println("Tree nr $i wird gefitted")
        # get random dataset of size forest.n_features_per_tree
        current_tree_features, current_tree_labels = get_random_features(features, labels, forest.n_features_per_tree)

        # TODO Regression or Classification?
        tree = DecisionTreeRegressor(max_depth=forest.max_depth)

        fit!(tree,current_tree_features, current_tree_labels)
        push!(forest.trees, tree)
    end
end

#TODO rename fit2, predict2 etc
function predict2(forest::ForestRegressor, X::Union{Matrix{S}, Vector{S}}) where S<:Union{Real, String}
    if isempty(forest.trees)
        error("Prediction failed because there are no trees. (Maybe you forgot to fit?)")
    end
    predictions = [predict(tree, X) for tree in forest.trees]
    return mean(predictions)
end