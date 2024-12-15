#This File contains the fundamentals for decision trees in Julia

# ----------------------------------------------------------------
# MARK: Structs & Constructors
# ----------------------------------------------------------------

"""
    DecisionFn

A structure representing a decision with a function and its parameter.

# Parameters
- `fn::Function`: The decision function.
- `param::Union{Real, String}`: The parameter for the decision function.
    - Real: for comparison functions (e.g. x < 5.0)
    - String: for True/False functions (e.g. x == "red" or x != 681)
"""
struct DecisionFn
    fn::Function
    param::Union{Real, String}
end

# """
#     Show representation of DecisionFn

# Displays the decision function.
# """
# Base.show(io::IO, ::MIME"text/plain", d::DecisionFn) = printDecisionFn(io, d)

# function printDecisionFn(io::IO, d::DecisionFn)
#     if isa(d.param, Real)
#         print(io, "x < ", d.param)
#     else
#         print(io, "x ", d.param)
#     end
# end

"""
    DecisionFn_to_string(d::DecisionFn)

Returns a string representation of the decision function.

# Arguments
- `d::DecisionFn`: The decision function to convert to a string.
"""

function DecisionFn_to_string(d::DecisionFn)
    if isa(d.param, Real)
        return "x < " * string(d.param)
    else
        return "x " * string(d.param)
    end
end


"""
    Node

A Node represents a decision in the Tree.
It is a leaf with a prediction or has exactly one true and one false child and a decision
function.
"""
struct Node
    decision::Union{DecisionFn, Nothing} # decision function
    true_child::Union{Node, Nothing} # decision is True
    false_child::Union{Node, Nothing} # decision is NOT true
    prediction::Union{Float64, Nothing} # for leaves
end

# Custom constructor for keyword arguments
function Node(; decision=nothing, true_child=nothing, false_child=nothing, prediction=nothing)
    Node(decision, true_child, false_child, prediction)
end


"""
    DecisionTree

A DecisionTree is a tree of Nodes.
In addition to a root node it holds meta informations such as max_depth etc.
Use `fit(tree, features, labels)` to create a tree from data

# Arguments
- root::Union{Node, Nothing}: the root node of the decision tree; `nothing` if the tree is empty
- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
struct DecisionTree
    root::Union{Node, Nothing}
    max_depth::Int

    # TODO: add additional needed properties here
    # min_samples_split::Int
    # pruning::Bool
    # rng=Random.GLOBAL_RNG

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
    #TODO: Implement CART
    error("Not implemented.")
end


"""
    build_tree(features, labels, max_depth, ...)

Builds a decision tree from the given data using some algorithm (e.g. CART)

# Arguments

- `tree::DecisionTree`: the tree to be trained
- `X::Array{Float64,2}`: the training data
- `y::Array{Float64,1}`: the target labels
"""
function build_tree(features::Array{Float64,2}, labels::Array{Float64,1},
                    max_depth::Int
                    #, min_samples_split::Int, pruning::Bool, rng=Random.GLOBAL_RNG
                    )::DecisionTree
    #TODO: Implement CART
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

A basic decision function for testing and playing around.
"""
function lessThan(x, threshold::Float64, featureindex::Int =1)::Bool
    return x[featureindex] < threshold
end


"""
    print_tree(tree::DecisionTree)

Prints a textual visualization of the decision tree.

# Arguments

- `tree`: The `DecisionTree` instance to print.

# Example output:

x < 28.0 ?
├─ False: y == 161.0 ?
│  ├─ False: 842
│  └─ True: 2493
└─ True: 683
"""
function print_tree(tree::DecisionTree)
    println(tree_to_string(tree))
end


"""
    tree_to_string(tree::DecisionTree)

Returns a textual visualization of the decision tree.
For each decision node, displays the condition, and for each leaf, displays the prediction.

# Arguments

- `tree`: The `DecisionTree` instance to print.
"""

function tree_to_string(tree::DecisionTree)
    if tree.root === nothing
        return "The tree is empty.\n"
    else
        # If leaf
        if tree.root.prediction !== nothing
            return "The tree is only a leaf with prediction = $(tree.root.prediction).\n"
        else
            result = "$(DecisionFn_to_string(tree.root.decision)) ?\n"
            result *= _node_to_string(tree.root.false_child, "", false, "")
            result *= _node_to_string(tree.root.true_child, "", true, "")
            return result
        end
    end
end

"""
    _node_to_string(node::Node, prefix::String, is_left::Bool, indentation::String)

Recursive helper function to stringify the decision tree structure.

# Arguments

- `node`: The current node to print.
- `prefix`: A string used for formatting the tree structure.
- `is_left`: Boolean indicating if the node is a left (true branch) child.
- `indentation`: The current indentation.
"""

function _node_to_string(node::Node, prefix::String, is_left::Bool, indentation::String)
    if is_left
        prefix = indentation * "└─ True"
    else
        prefix = indentation * "├─ False"
    end
    # If leaf
    if node.prediction !== nothing
        return "$(prefix): $(node.prediction)\n"
    else
        result = "$(prefix): $(DecisionFn_to_string(node.decision)) ?\n"
        if is_left
            indentation = indentation * "   "
        else
            indentation = indentation * "│  "
        end
        result *= _node_to_string(node.false_child, prefix, false, indentation)
        result *= _node_to_string(node.true_child, prefix, true, indentation)
        return result
    end
end
