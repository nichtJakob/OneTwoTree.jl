#This File contains the fundamentals for decision trees in Julia

include("DecisionFunction.jl")

# ----------------------------------------------------------------
# MARK: Structs & Constructors
# ----------------------------------------------------------------

"""
    Node

A Node represents a decision in the Tree.
It is a leaf with a prediction or has exactly one true and one false child and a decision
function.
"""
mutable struct Node{S<:Union{Real, String}, T<:Union{Number, String}}
    # Reference to whole dataset governed by the tree (This is not a copy as julia doesn't copy but only binds new aliases to the same object)
    # data points are rows, data features are columns
    dataset::Union{Matrix{S}, Nothing}
    # labels can be categorical => String or numerical => Real
    labels::Union{Vector{T}, Nothing}
    # Indices of the data in the dataset being governed by this node
    node_data::Vector{Int64}
    # TODO: Index list of constant columns or columns the label does not vary with
    # constant_columns::Vector{Int64}
    # Own impurity
    impurity::Union{Float64, Nothing}
    depth::Int64

    # TODO: should implement split_function; split function should only work if this node is a leaf
    # decision::Union{Function, Nothing} #returns True -> go to right child else left
    decision::Union{Decision, Nothing} #returns True -> go to right child else left
    decision_string::Union{String, Nothing} # *Optional* string for printing

    true_child::Union{Node, Nothing} #decision is True
    false_child::Union{Node, Nothing} #decision is NOT true
    prediction::Union{T, Nothing} # for leaves

    # Constructor handling assignments & splitting
    # TODO: replace classify::Bool with enum value for readability
    function Node(dataset::Matrix{S}, labels::Vector{T}, node_data::Vector{Int64}, classify::Bool; depth=0, min_purity_gain=nothing, max_depth=0) where {S, T}
        N = new{S, T}(dataset, labels, node_data)
        N.depth = depth
        N.true_child = nothing
        N.false_child = nothing

        # Determine the best prediction in this node if it is/were a leaf node
        # (We calculate the prediction even in non-leaf nodes, because we need it to decide whether to split this node. This is because we also consider how much purity is gained by splitting this node.)
        if classify
            # in classification, we simply choose the most frequent label as our prediction
            N.prediction = most_frequent_class(labels, node_data)
            # calculate gini impurity if this was a leaf node
            N.impurity = gini_impurity(dataset, labels, node_data)
        else
            # in regression, we choose the mean as our prediction as it minimizes the square loss
            N.prediction = label_mean(labels, node_data)
            N.impurity = 0.65 # TODO: in regression Sum-of-squares error is used as measure of impurity
        end

        N.decision, post_split_impurity = split(N)
        if should_split(N, post_split_impurity, max_depth)
            # N.decision_column = split_info...
            # Partition dataset into true/false datasets & pass them to the children
            true_data, false_data = split_indices(N.dataset, N.node_data, N.decision.fn, N.decision.param, N.decision.feature)
            N.true_child = Node(dataset, labels, true_data, classify, depth=N.depth+1, min_purity_gain=min_purity_gain, max_depth=max_depth)
            N.false_child = Node(dataset, labels, false_data, classify, depth=N.depth+1, min_purity_gain=min_purity_gain, max_depth=max_depth)
            # TODO: Do we want to set prediction to nothing in non-leaf nodes? It could be neat to just have it, if we already had to calculate it anyways.
            N.prediction = nothing
        else
            # Clear decision as we don't want to split
            N.decision = nothing
        end
        return N
    end
end

# Custom constructor for keyword arguments
function Node(dataset, labels, classify; column_data=false, node_data=nothing, max_depth=0)

    # This is meant for when initializing a matrix like [[] [] []]. Then the inner []'s are inserted into the matrix as column vectors.
    # But since we would like them to be interpreted as row-vectors, we provide the option to transpose in this case.
    if column_data == true
        dataset = copy(transpose(dataset))
    end
    # if no subset was passed
    if node_data === nothing
        node_data = collect(1:size(dataset, 1))
    end
    return Node(dataset, labels, node_data, classify, max_depth=max_depth)
end

# function Node(dataset; node_data=nothing, decision=nothing, true_child=nothing, false_child=nothing, prediction=nothing)
    # Node(dataset, decision, true_child, false_child, prediction)
# end

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
Some guards to ensure the input data is valid for training a tree.
"""
function _verify_fit!_args(tree, dataset, labels, column_data)
    if isempty(labels)
        error("fit!: Cannot build tree from empty label set.")
    end
    if isempty(dataset)
        error("fit!: Cannot build tree from empty dataset.")
    end
    if tree.max_depth < -1
        error("fit!: Cannot build tree with negative depth, but got max_depth=$(max_depth).")
    end
    if (!column_data && size(dataset, 1) != length(labels))
        error("fit!: Dimension mismatch! Number of datapoints $(size(dataset, 1)) != number of labels $(length(labels)).\n Maybe transposing your dataset matrix or setting column_data=true helps?")
    end
    if (column_data && size(dataset, 2) != length(labels))
        error("fit!: Dimension mismatch! Number of datapoints $(size(dataset, 2)) != number of labels $(length(labels)).\n Maybe transposing your dataset matrix or setting column_data=false helps?")
    end
    for label in labels
        if typeof(label) != typeof(labels[1])
            error("fit!: Encountered heterogeneous label types. Please make sure all labels are of the same type.")
        end
    end
    if tree isa DecisionTreeRegressor && (labels[1] isa String) # vorher: !(labels[1] isa String)
        error("Cannot train a DecisionTreeRegressor on a dataset with categorical labels.")
    end

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
end

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
    _verify_fit!_args(tree, features, labels, column_data)

    classify = (tree isa DecisionTreeClassifier)
    tree.root = Node(features, labels, classify, max_depth=tree.max_depth, column_data=column_data)
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

"""
    is_leaf(node)

Do you seriously expect a description for this?
"""
function is_leaf(node::Node)::Bool
    return node.prediction !== nothing
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


"""
    _node_to_string(node::Node, prefix::String, is_true_child::Bool, indentation::String)

Recursive helper function to stringify the decision tree structure.

# Arguments

- `node`: The current node to print.
- `is_true_child`: Boolean indicating if the node is a true branch child.
- `indentation`: The current indentation.
"""
function _node_to_string(node::Node, is_true_child::Bool, indentation::String)
    if is_true_child
        prefix = indentation * "├─ True"
    else
        prefix = indentation * "└─ False"
    end

    if is_leaf(node)
        return "$(prefix): $(node.prediction)\n"
    else
        result = "$(prefix): $(node.decision) ?\n"
        if is_true_child
            indentation = indentation * "   "
        else
            indentation = indentation * "│  "
        end
        result *= _node_to_string(node.true_child, true, indentation)
        result *= _node_to_string(node.false_child, false, indentation)
        return result
    end
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