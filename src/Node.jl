include("DecisionFunction.jl")

"""
    Node

A Node represents a decision in the Tree.
It is a leaf with a prediction or has exactly one true and one false child and a decision
function.
"""
mutable struct Node{T<:Union{Number, String}}
    # Reference to whole dataset governed by the tree (This is not a copy as julia doesn't copy but only binds new aliases to the same object)
    # data points are rows, data features are columns
    dataset::Union{AbstractMatrix, Nothing}
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
    function Node(dataset::AbstractMatrix, labels::Vector{T}, node_data::Vector{Int64}, classify::Bool; depth=0, min_purity_gain=nothing, max_depth=0) where {S, T}
        N = new{T}(dataset, labels, node_data)
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
            # NOTE: The reason it is set to nothing here atm, is because N.prediction being nothing is later used to identify non-leaf nodes.
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

"""
    is_leaf(node)

Do you seriously expect a description for this?
"""
function is_leaf(node::Node)::Bool
    return node.prediction !== nothing
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

    if node === nothing
        return "$(prefix): <Nothing>\n"
    end
    if is_leaf(node)
        return "$(prefix): $(node.prediction)\n"
    end

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

function _node_to_string_as_root(node::Node)
    if is_leaf(node)
        return "\nPrediction: $(node.prediction)\n"
    end

    result = "\n$(node.decision) ?\n"
    result *= _node_to_string(node.true_child, true, "")
    result *= _node_to_string(node.false_child, false, "")
    return result
end

function Base.show(io::IO, node::Node)
    print(io, _node_to_string_as_root(node))
end