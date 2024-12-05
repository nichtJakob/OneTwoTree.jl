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
struct Node
    decision::Union{Function, Nothing} #returns True -> go to right child else left
    true_child::Union{Node, Nothing} #decision is True
    false_child::Union{Node, Nothing} #decision is NOT true
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

- `max_depth::Int`: maximum depth of the decision tree; no limit if equal to -1
"""
function DecisionTree(; max_depth=-1)
    DecisionTree(nothing, max_depth)
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


