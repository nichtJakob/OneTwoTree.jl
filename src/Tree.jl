#This File contains the fundamentals for decision trees in Julia

# ----------------------------------------------------------------
# MARK: Structs & Constructors
# ----------------------------------------------------------------

#MARK: DecisionTree
abstract type AbstractDecisionTree end

"""
    DecisionTreeClassifier

A DecisionTreeClassifier is a tree of decision nodes. It can predict classes based on the input data.
In addition to a root node it holds meta informations such as max_depth etc.
Use `fit(tree, features, labels)` to create a tree from data

# Arguments
- root::Union{Node, Nothing}: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
mutable struct DecisionTreeClassifier <: AbstractDecisionTree
    root::Union{Node, Nothing}
    max_depth::Int
end

"""
    Initialises a decision tree model.

# Arguments

- `root::Union{Node, Nothing}`: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
function DecisionTreeClassifier(; root=nothing, max_depth=-1)
    if max_depth < -1
        error("DecisionTreeClassifier: Got invalid max_depth. Set it to a value >= -1. (-1 means unlimited depth)")
    end
    DecisionTreeClassifier(root, max_depth)
end

"""
    DecisionTreeRegressor

A DecisionTreeRegressor is a tree of decision nodes. It can predict function values based on the input data.
In addition to a root node it holds meta informations such as max_depth etc.
Use `fit(tree, features, labels)` to create a tree from data

# Arguments
- root::Union{Node, Nothing}: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
mutable struct DecisionTreeRegressor <: AbstractDecisionTree
    root::Union{Node, Nothing}
    max_depth::Int
end

"""
    Initialises a decision tree model.

# Arguments

- `root::Union{Node, Nothing}`: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
function DecisionTreeRegressor(; root=nothing, max_depth=-1)
    if max_depth < -1
        error("DecisionTreeRegressor: Got invalid max_depth. Set it to a value >= -1. (-1 means unlimited depth)")
    end
    DecisionTreeRegressor(root, max_depth)
end



# ----------------------------------------------------------------
# MARK: Functions
# ----------------------------------------------------------------

"""
    fit!(tree, features, labels)

Train a decision tree on the given data using some algorithm (e.g. CART).

# Arguments

- `tree::AbstractDecisionTree`: the tree to be trained
- `dataset::Matrix{Union{Real, String}}`: the training data
- `labels::Vector{Union{Real, String}}`: the target labels
- `column_data::Bool`: whether the datapoints are contained in dataset columnwise
"""
function fit!(tree::AbstractDecisionTree, features::Matrix{S}, labels::Vector{T}, column_data=false) where {S<:Union{Real, String}, T<:Union{Number, String}}
    if isempty(labels)
        error("Cannot build tree from empty label set.")
    end
    if tree isa DecisionTreeRegressor && (labels[1] isa String) # vorher: !(labels[1] isa String)
        error("Cannot train a DecisionTreeRegressor on a dataset with categorical labels.")
    end

    classify = (tree isa DecisionTreeClassifier)
    tree.root = Node(features, labels, classify, max_depth=tree.max_depth, column_data=column_data)
end


"""
    build_tree(features, labels, max_depth, ...)

Builds a decision tree from the given data using some algorithm (e.g. CART)

# Arguments

- `tree::AbstractDecisionTree`: the tree to be trained
- `dataset::Matrix{Union{Real, String}}`: the training data
- `labels::Vector{Union{Real, String}}`: the target labels
- `max_depth::Int`: the maximum depth of the created tree
- `column_data::Bool`: whether the datapoints are contained in dataset columnwise
"""
function build_tree(dataset::Matrix{S}, labels::Vector{T},
                    max_depth::Int;
                    column_data=false
                    #, min_samples_split::Int, pruning::Bool
                    )::AbstractDecisionTree where {S<:Union{Real, String}, T<:Union{Real, String}}

    # TODO: probably move these checks to a dedicated consistency function
    if isempty(labels)
        error("build_tree: Cannot build tree from empty label set.")
    end
    if isempty(dataset)
        error("build_tree: Cannot build tree from empty dataset.")
    end
    if max_depth < 0
        error("build_tree: Cannot build tree with negative depth, but got max_depth=$(max_depth).")
    end
    if (!column_data && size(dataset, 1) != length(labels))
        error("build_tree: Dimension mismatch! Number of datapoints $(size(dataset, 1)) != number of labels $(length(labels)).\n Maybe transposing your dataset matrix or setting column_data=true helps?")
    end
    if (column_data && size(dataset, 2) != length(labels))
        error("build_tree: Dimension mismatch! Number of datapoints $(size(dataset, 2)) != number of labels $(length(labels)).\n Maybe transposing your dataset matrix or setting column_data=false helps?")
    end
    for label in labels
        if typeof(label) != typeof(labels[1])
            error("build_tree: Encountered heterogeneous label types. Please make sure all labels are of the same type.")
        end
    end

    # TODO:
    # TODO: check if columns of dataset have consistent type either Real or String
    # if !column_data
    #     for i in range(1, size(dataset, 2))
    #         for j in range(1, size(dataset, 1))
    #             if typeof(dataset[j, i]) != typeof(dataset[1, i])
    #                 error("build_tree: Encountered heterogeneous feature types. Please make sure matching features of all datapoints have the same type.")
    #             end
    #         end
    #     end
    # end

    # if column_data
    #     for i in range(1, size(dataset, 1))
    #         for j in range(1, size(dataset, 2))
    #             if typeof(dataset[i, j]) != typeof(dataset[i, 1])
    #                 error("build_tree: Encountered heterogeneous feature types. Please make sure matching features of all datapoints have the same type.")
    #             end
    #         end
    #     end
    # end

    classify = (labels[1] isa String)
    # root = Node(dataset, labels, classify, max_depth=max_depth, column_data=column_data)
    if(classify)
        tree = DecisionTreeClassifier(nothing, max_depth)
    else
        tree = DecisionTreeRegressor(nothing, max_depth)
    end
    fit!(tree, dataset, labels, column_data=column_data)
    # TODO: pruning
    return tree
end

"""
    predict

Traverses the tree for a given datapoint x and returns that trees prediction.

# Arguments

- `tree::AbstractDecisionTree`: the tree to predict with
- `X::Union{Matrix{S}, Vector{S}`: the data to predict on
"""
function predict(tree::AbstractDecisionTree, X::Union{Matrix{S}, Vector{S}}) where S<:Union{Real, String}
    if tree.root === nothing
        error("Cannot predict from an empty tree.")
    end

    return predict(tree.root, X)
end

function predict(node::Node, datapoint::Vector{S}) where S<:Union{Real, String}
    if is_leaf(node)
        return node.prediction
    end

    if call(node.decision, datapoint)
        return predict(node.true_child, datapoint)
    else
        return predict(node.false_child, datapoint)
    end
end

function predict(node::Node, dataset::Matrix{S}) where S<:Union{Real, String}
    if is_leaf(node)
        return node.prediction * ones(size(dataset, 1))
    end

    result = []

    for i in range(1, size(dataset, 1))
        datapoint = dataset[i, :]
        if call(node.decision, datapoint)
            push!(result, predict(node.true_child, datapoint))
        else
            push!(result, predict(node.false_child, datapoint))
        end
    end
    return result
end

"""
    calc_accuracy(labels, predictions)

Calculates the accuracy of the predictions compared to the labels.
"""
function calc_accuracy(labels, predictions)
    if length(labels) != length(predictions)
        error("Length of labels and predictions must be equal.")
    end

    if length(labels) == 0
        return 0.0
    end

    correct = 0.0
    for i in 1:lastindex(labels)
        if labels[i] == predictions[i]
            correct += 1.0
        end
    end

    return correct / length(labels)
end

"""
    depth(tree)

Traverses the tree and returns the maximum depth.
"""
function calc_depth(tree::AbstractDecisionTree)

    max_depth = 0
    if tree.root === nothing
        return max_depth
    end

    to_visit = [(tree.root, 0)]
    while !isempty(to_visit)
        node, cur_depth = popfirst!(to_visit)

        if cur_depth > max_depth
            max_depth = cur_depth
        end

        if node.true_child !== nothing
            push!(to_visit, (node.true_child, cur_depth + 1))
        end

        if node.false_child !== nothing
            push!(to_visit, (node.false_child, cur_depth + 1))
        end
    end
    return max_depth
end

#----------------------------------------
# MARK: Printing
#----------------------------------------

"""
    tree_to_string(tree::AbstractDecisionTree)

Returns a textual visualization of the decision tree.

# Arguments

- `tree::AbstractDecisionTree` The `DecisionTree` instance to print.

# Example output:

x < 28.0 ?
├─ False: x == 161.0 ?
│  ├─ False: 842
│  └─ True: 2493
└─ True: 683
"""
function _tree_to_string(tree::AbstractDecisionTree)
    if tree.root === nothing
        return "\n<Empty Tree>\n"
    end

    if is_leaf(tree.root)
        return "\nRoot Prediction: $(tree.root.prediction).\n"
    end

    result = "\n$(tree.root.decision) ?\n"
    result *= _node_to_string(tree.root.true_child, true, "")
    result *= _node_to_string(tree.root.false_child, false, "")
    return result
end

function Base.show(io::IO, tree::AbstractDecisionTree)
    print(io, _tree_to_string(tree))
end

"""
    print_tree(tree::AbstractDecisionTree)

Returns a textual visualization of the decision tree.

# Arguments

- `tree::AbstractDecisionTree` The `DecisionTree` instance to print.

# Example output:

x < 28.0 ?
├─ False: x == 161.0 ?
│  ├─ False: 842
│  └─ True: 2493
└─ True: 683
"""
function print_tree(tree::AbstractDecisionTree)
    print(tree)
end