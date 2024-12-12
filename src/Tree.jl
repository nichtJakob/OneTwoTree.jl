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
    decision::Union{Function, Nothing} # returns True -> go to left child, else right
    decision_string::Union{String, Nothing} # *Optional* string for printing
    true_child::Union{Node, Nothing} #decision is True
    false_child::Union{Node, Nothing} #decision is NOT true
    prediction::Union{Float64, Nothing} # for leaves
end

# Custom constructor for keyword arguments
function Node(; decision=nothing, decision_string=nothing, true_child=nothing, false_child=nothing, prediction=nothing)
    Node(decision, decision_string, true_child, false_child, prediction)
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
