#This File contains the fundamentals for decision trees in Julia



"""
    Node

A Node represents a decision in the Tree.
It is a leaf with a prediction or has exactly one true and one false child and a decision function.
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
    tree_prediction

Traverses the tree for a given datapoint x and returns that trees prediction
"""
function tree_prediction(tree::Node, x)
    #Check if leaf
    if tree.prediction != nothing
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

a basic decision function for testing and playing around
"""
function lessThan(x, threshold::Float64, featureindex::Int =1)::Bool
    return x[featureindex] < threshold
end
