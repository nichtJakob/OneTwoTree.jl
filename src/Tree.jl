#This File contains the fundamentals for decision trees in Julia

# ----------------------------------------------------------------
# MARK: Structs & Constructors
# ----------------------------------------------------------------

"""
    Node

A Node represents a decision in the Tree.
It is a leaf with a prediction or has exactly one true and one false child and a decision
function.
"""
mutable struct Node{S<:Union{Real, String}, T<:Union{Real, String}}
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
    decision::Union{Function, Nothing} #returns True -> go to right child else left
    decision_string::Union{String, Nothing} # *Optional* string for printing

    true_child::Union{Node, Nothing} #decision is True
    false_child::Union{Node, Nothing} #decision is NOT true
    prediction::Union{T, Nothing} # for leaves

    # Constructor handling assignments & splitting
    # TODO: replace classify::Bool with enum value for readability
    function Node(dataset::Matrix{S}, labels::Vector{T}, node_data::Vector{Int64}, classify::Bool; depth=0, min_purity_gain=nothing, max_depth=0) where {S, T}
        N = new{S, T}(dataset, labels, node_data)
        N.depth = depth

        # Determine the best prediction in this node if it is/were a leaf node
        # (We calculate the prediction even in non-leaf nodes, because we need it to decide whether to split this node. This is because we also consider how much purity is gained by splitting this node.)
        if classify
            # in classification, we simply choose the most frequent label as our prediction
            N.prediction = most_frequent_class(labels, node_data)
        else
            # in regression, we choose the mean as our prediction as it minimizes the square loss
            N.prediction = label_mean(labels, node_data)
        end

        N.impurity = 0.65 # TODO: dummy impurity; remove once the actual impurity can be calculated
        # TODO: N.impurity = gini_impurity(dataset, labels, node_data) # TODO: create gini_impurity function that works with an index list
        # N.impurity = gini_impurity(dataset[node_data, :], labels[node_data, :]) # This version uses the index list to index into the dataset before calling the method
        # TODO: gini_impurity also needs to support non true/false labels (As it is, I can't use it right now)

        # TODO: This is where the actual work is at
        # split_info = calc_best_split()
        split(N)
        if should_split(N, max_depth)
            # N.decision_column = split_info...
            # N.decision = ...
            # Partition dataset into true/false datasets & pass them to the children
            # N.true_child = ...
            # N.false_child = ...

        end
        return N
    end

end

# Custom constructor for keyword arguments
function Node(dataset, labels, classify; node_data=nothing, max_depth=0)

    # if no subset was passed
    if node_data == nothing
        node_data = collect(1:size(dataset, 1))
    end
    return Node(dataset, labels, node_data, classify)
end

# function Node(dataset; node_data=nothing, decision=nothing, true_child=nothing, false_child=nothing, prediction=nothing)
    # Node(dataset, decision, true_child, false_child, prediction)
# end

function split(N::Node)
    # 1. find best feature to split i.e. calc best split for each feature
    num_features = size(N.dataset)[2]
    best_feature = -1
    # best_split_threshold = nothing
    # best_split_choice = nothing
    best_split_function = nothing
    best_impurity = -1

    for i in range(1, num_features)
        # NOTE: This determination of whether a column is categorical or numerical assumes, that the types do not vary among a column
        is_categorical = (typeof(N.dataset[1, i]) == String)
        if is_categorical
            classes = collect_classes(N.dataset, i)
            for class in classes

                # TODO: impurity = gini_impurity()
                impurity = 0.65

                if best_feature == -1 || (impurity < best_impurity)
                    best_feature = i
                    best_impurity = impurity
                    # TODO: set correct split function: best_split_function = equal(, i, class)
                end
            end
        else
            # sort dataset matrix by column
            feature_value_sorting = sortperm(N.dataset[:, i])
            for j in range(1, size(feature_value_sorting)[1]-1)
                value = N.dataset[feature_value_sorting[j]]
                next_value = N.dataset[feature_value_sorting[j+1]]
                midpoint =  (value + next_value)/2.0

                # TODO: actually calculate impurity
                impurity = 0.65

                if best_feature == -1 || (impurity < best_impurity)
                    best_feature = i
                    best_impurity = impurity
                    # TODO: set correct split function: best_split_function = lessThan(, i, threshold)
                end
            end
        end
    end

    # there are sortperm() and sortrows() for this
    # or maybe its best to use dataframes, which offer this functionality as well
    # => copy and sort? Or use index-permutations to quantify order?

    # for numerical features, we sort them and calculate the gini impurity for each split (splitting at the mean between each two list neighbors)
    # for categorical features, we calculate the gini impurity for each split (e.g. feature == class1, feature == class2, ...)

    # Then we choose the overall best split we found

    # best_split = nothing
    # foreach feature
    #     foreach possible split
    #         calculate splitting criterion e.g. gini impurity
    #         if this_split is better than best_split (according to criterion)
    #             best_split = this_split
    # return best_split
end

function should_split(N::Node, max_depth::Int64)
    # TODO: implement actual splitting decision logic i.e. do we want to split this node yey or nay?
    # There are a variety of criteria one could imagine. For now we only posit that the current node should be impure i.e. impurity > 0 and the max_depth hasn't been reached.
    if N.impurity == 0.0
        return false
    end
    if N.depth == max_depth
      return false
    end
    # if impurity - impurity_after_split < min_purity_gain
    #   return false
    # end

    return true
end

# split node_data indices according to decision function
function split_indices(dataset::Matrix{S}, node_data::Vector{Int64}, decision_fn::Function) where S
    true_child_data::Vector{Int64} = []
    false_child_data::Vector{Int64} = []
    for datapoint_idx in node_data
        if decision_fn(dataset[datapoint_idx, :])
            push!(true_child_data, datapoint_idx)
        else
            push!(false_child_data, datapoint_idx)
        end
    end
    return true_child_data, false_child_data
end

function collect_classes(dataset::Matrix{S}, column::Int64) where S
    classes = Dict{String, Bool}()
    # TODO: check if passed column is out of bounds
    rows = size(dataset)[1]
    for i in range(1, rows)
        value = dataset[i, column]
        if !haskey(classes, value)
            classes[value] = true
        end
    end
    return collect(keys(classes))
end

function most_frequent_class(labels::Vector{String}, indices::Vector{Int64})
    class_frequencies = Dict{String, Int64}()
    most_frequent = nothing

    for index in indices
        class = labels[index]
        if haskey(class_frequencies, class)
            class_frequencies[class] = class_frequencies[class] + 1
            if class_frequencies[class] > class_frequencies[most_frequent]
                most_frequent = class
            end
        else
            class_frequencies[class] = 1
            if most_frequent == nothing
                most_frequent = labels[index]
            end
        end
    end
    return most_frequent
end

function label_mean(labels::Vector{T}, indices::Vector{Int64}) where T<:Real
    sum = 0.0
    foreach(index -> sum += labels[index], indices)
    return sum / size(labels[indices])[1]
end

function label_mean(labels::Vector{T}) where T<:Real
    sum = 0.0
    foreach(label -> sum += label, labels)
    return sum / size(labels)[1]
end

"""
    DecisionTree

A DecisionTree is a tree of Nodes.
In addition to a root node it holds meta informations such as max_depth etc.
Use `fit(tree, features, labels)` to create a tree from data

# Parameters
- root::Union{Node, Nothing}: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
struct DecisionTree
    root::Union{Node, Nothing}
    max_depth::Int

    # TODO: add additional needed properties here
    # min_samples_split::Int
    # pruning::Bool

    # default constructor
    function DecisionTree(root::Union{Node, Nothing}, max_depth::Int)
        new(root, max_depth)
    end
end

"""
    Initialises a decision tree model.

# Arguments

- `root::Union{Node, Nothing}`: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""


function DecisionTree(; root=nothing, max_depth=-1)
    DecisionTree(root, max_depth)
end

# ----------------------------------------------------------------
# MARK: Functions
# ----------------------------------------------------------------

"""
    fit!(tree, features, labels)

Train a decision tree on the given data using some algorithm (e.g. CART).

# Arguments

- `tree::DecisionTree`: the tree to be trained
- `X::Array{Float64,2}`: the training data
- `y::Array{Float64,1}`: the target labels
"""
function fit!(tree::DecisionTree, features::Array{Float64,2}, labels::Array{Float64,1})
# TODO: function fit!(tree::DecisionTree, dataset::Matrix{S}, labels::Vector{T}) <: {S, T}
    #TODO: Implement CART
    error("Not implemented.")
end


"""
    build_tree(features, labels, max_depth, ...)

Builds a decision tree from the given data using some algorithm (e.g. CART)

# Arguments

- `tree::DecisionTree`: the tree to be trained
- `X::Array{Float64,2}`: the training data
- `y::Array{String,1}`: the target labels
- `max_depth::Int`: maximum depth of the created tree
"""
# TODO: overload this function by a regression version with y::Vector{Float64} and a classification version with y::Vector{String} and remove 
function build_tree(features::Array{Float64,2}, labels::Array{String,1},
                    max_depth::Int, classify::Bool
                    #, min_samples_split::Int, pruning::Bool
                    )::DecisionTree
    #TODO: Implement CART
    # TODO: build tree via root node
    # TODO: pruning
    error("Not implemented.")
end


"""
    tree_prediction

Traverses the tree for a given datapoint x and returns that trees prediction.
"""
function tree_prediction(tree::Node, x)
    #Check if leaf
    if tree.prediction !== nothing
        return tree.prediction
    end

    #else check if decision(x) leads to right or left child
    if tree.decision(x)
        return tree_prediction(tree.true_child, x)
    else
        return tree_prediction(tree.false_child, x)
    end
end


"""
    lessThan

A basic numerical decision function for testing and playing around.
"""
function lessThan(x, threshold::Float64, featureindex::Int = 1)::Bool
    return x[featureindex] < threshold
end

"""
    equal

A basic categorical decision function for testing and playing around.
"""
function equal(x, class::String, featureindex::Int = 1)::Bool
    return x[featureindex] == class
end

"""
    print_tree(tree::DecisionTree)

Prints a textual visualization of the decision tree.
For each decision node, it displays the condition, and for each leaf, it displays the prediction.

# Arguments

- `tree`: The `DecisionTree` instance to print.

# Example output:

x < 28 ?
├─ False: y < 161 ?
   ├─ False: 842
   └─ True: 2493
└─ True: 683

"""

function print_tree(tree::DecisionTree)
    if tree.root === nothing
        println("The tree is empty.")
    else
        # If leaf
        if tree.root.prediction !== nothing
            println("The tree is only a leaf with prediction = ", tree.root.prediction, ".")
        else
            println(string(tree.root.decision_string), " ?")
            _print_node(tree.root.true_child, "", false, "")
            _print_node(tree.root.false_child, "", true, "")
        end
    end
end

"""
    _print_node(node::Node, prefix::String, is_left::Bool, indentation::String)

Recursive helper function to print the decision tree structure.

# Arguments

- `node`: The current node to print.
- `prefix`: A string used for formatting the tree structure.
- `is_left`: Boolean indicating if the node is a left (true branch) child.
- `indentation`: The current indentation.
"""

function _print_node(node::Node, prefix::String, is_left::Bool, indentation::String)
    if is_left
        prefix = indentation * "└─ True"
    else
        prefix = indentation * "├─ False"
    end
    # If leaf
    if node.prediction !== nothing
        println(prefix, ": ", node.prediction)
    else
        println(prefix, ": ", string(tree.root.decision_string), " ?")
        _print_node(node.true_child, prefix, false, indentation * "   ")
        _print_node(node.false_child, prefix, true, indentation * "   ")
    end
end
