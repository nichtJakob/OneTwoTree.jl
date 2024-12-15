#This File contains the fundamentals for decision trees in Julia

# ----------------------------------------------------------------
# MARK: Structs & Constructors
# ----------------------------------------------------------------

struct Decision{S<:Union{Real, String}}
    fn::Function
    param::S
    feature::Int64

    function Decision(fn::Function, feature::Int64, param::S) where S
        # TODO: feature index can be chosen out of bounds... Idk, just be careful?
        new{S}(fn, param, feature)
    end
end

function call(decision::Decision, datapoint::Vector{S}) where S
    if length(datapoint) < decision.feature
        error("call: passed datapoint of insufficient dimensionality!")
    end
    return decision.fn(datapoint, decision.param, feature=decision.feature)
end

function call(decision::Decision, dataset::Matrix{S}) where S
    if size(dataset, 2) < decision.feature
        error("call: passed dataset with data of insufficient dimensionality!")
    end
    return [decision.fn(datapoint, decision.param, feature=decision.feature) for datapoint in dataset]
end

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

function split(N::Node)
    decision::Union{Decision, Nothing} = nothing

    # 1. find best feature to split i.e. calc best split for each feature
    num_features = size(N.dataset)[2]
    best_feature = -1
    best_decision::Union{Decision, Nothing} = nothing
    best_impurity = -1.0

    data = N.dataset[N.node_data, :]
    for i in range(1, num_features)
        # NOTE: This determination of whether a column is categorical or numerical assumes, that the types do not vary among a column
        is_categorical = (typeof(N.dataset[1, i]) == String)
        # @info "\n\n\nChecking decisions for feature $(i) where is_categorical=$(is_categorical): "
        # for categorical features, we calculate the gini impurity for each split (e.g. feature == class1, feature == class2, ...)
        if is_categorical
            # TODO: Test & Debug Categorical case
            # TODO: collect from data not N.dataset or write another collect_classes that takes a node_data index list as well
            classes = collect_classes(N.dataset, i)
            for class in classes

                decision = Decision(equal, i, class)
                impurity = gini_impurity(N.dataset, N.labels, N.node_data, decision.fn, decision.param, decision.feature)

                if best_feature == -1 || (impurity < best_impurity)
                    best_feature = i
                    best_impurity = impurity
                    best_decision = decision
                end
            end
        # for numerical features, we sort them and calculate the gini impurity for each split (splitting at the mean between each two list neighbors)
        else
            # sort dataset matrix by column
            feature_value_sorting = sortperm(data[:, i])
            j = 1
            while j < length(feature_value_sorting)
                value = data[feature_value_sorting[j], i]
                next_value = data[feature_value_sorting[j+1], i]
                # if next_value == value there is no discriminating decision,
                # thus we forward to the next distinct value
                while next_value == value
                    j += 1
                    if j < size(feature_value_sorting)[1]
                        next_value = data[feature_value_sorting[j+1], i]
                    else
                        j = -1
                        break
                    end
                end
                if j == -1
                    break
                end
                # calculate threshold used to discriminate between two values
                midpoint = (value + next_value)/2.0

                # calculate splitting impurity
                decision = Decision(lessThanOrEqual, i, midpoint)
                impurity = gini_impurity(N.dataset, N.labels, N.node_data, decision.fn, decision.param, decision.feature)

                # check if we found an improving decision
                if best_feature == -1 || (impurity < best_impurity)
                    best_feature = i
                    best_impurity = impurity
                    best_decision = decision
                end
                j += 1
            end
        end
    end

    # if best_decision == nothing, this means that no split could be found.
    return best_decision, best_impurity
end

function should_split(N::Node, post_split_impurity::Float64, max_depth::Int64)
    # TODO: implement actual splitting decision logic i.e. do we want to split this node yey or nay?
    # There are a variety of criteria one could imagine. For now we only posit that the current node should be impure i.e. impurity > 0 and the max_depth hasn't been reached.
    if N.decision === nothing || post_split_impurity == -1.0
        # @info "Could not find optimal split => No Split"
        return false
    end
    if N.impurity == 0.0
        # @info "Node impurity == 0.0 => No Split"
        return false
    end
    if N.depth == max_depth
        # @info "max_depth has been reached => No Split"
      return false
    end
    # if impurity - post_split_impurity < min_purity_gain
    #   return false
    # end
    return true
end

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

"""
    lessThanOrEqual

A basic numerical decision function for testing and playing around.
"""
function lessThanOrEqual(x, threshold::Float64; feature::Int64 = 1)::Bool
    return x[feature] <= threshold
end

"""
    equal

A basic categorical decision function for testing and playing around.
"""
function equal(x, class::String; feature::Int64 = 1)::Bool
    return x[feature] == class
end

"""
    print_tree(tree::AbstractDecisionTree)

Prints a textual visualization of the decision tree.
For each decision node, it displays the condition, and for each leaf, it displays the prediction.

# Arguments

- `tree::AbstractDecisionTree` The `DecisionTree` instance to print.

# Example output:

x < 28 ?
├─ False: y < 161 ?
   ├─ False: 842
   └─ True: 2493
└─ True: 683

"""
function print_tree(tree::AbstractDecisionTree)
    if tree.root === nothing
        println("The tree is empty.")
    else
        # TODO: You cannot decide whether a node is a leaf or not by whether it has a predictiona associated with it.
        # If leaf
        # if tree.root.prediction !== nothing
        #     println("The tree is only a leaf with prediction = ", tree.root.prediction, ".")
        # else
        # TODO: please don't use assumed to be pre-stored decision strings, and calculate them yourself
        # println(string(tree.root.decision_string), " ?")
        println("DECISION: $(tree.root.decision)")
        _print_node(tree.root.true_child, "", true, "")
        _print_node(tree.root.false_child, "", false, "")
        # end
    end
end

"""
    is_leaf(node)

Do you seriously expect a description for this?
"""
function is_leaf(node::Node)::Bool
    return node.prediction !== nothing
end

"""
    _print_node(node::Node, prefix::String, is_left::Bool, indentation::String)

Recursive helper function to print the decision tree structure.

# Arguments

- `node`: The current node to print.
- `prefix`: A string used for formatting the tree structure.
- `is_true_child`: Boolean indicating if the node is a true branch child.
- `indentation`: The current indentation.
"""

function _print_node(node::Node, prefix::String, is_true_child::Bool, indentation::String)
    if is_true_child
        prefix = indentation * "├─ True"
    else
        prefix = indentation * "└─ False"
    end
    # If leaf
    # TODO: You cannot decide whether a node is a leaf or not by whether it has a predictiona associated with it.
    if node.prediction !== nothing
        println(prefix, ": ", node.prediction)
    else
        # println(prefix, ": ", string(tree.root.decision_string), " ?")
        println("DECISION: $(node.decision)")
        _print_node(node.true_child, prefix, true, indentation * "   ")
        _print_node(node.false_child, prefix, false, indentation * "   ")
    end
end