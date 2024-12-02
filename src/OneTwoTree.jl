module OneTwoTree


"""
    Node

A Node represents a decision in the Tree.
It has at most one left and one right child.
"""
struct Node
    decision_is_right::Union{Function, Nothing} #returns True -> go to right child else left
    left_child::Union{Node, Nothing} #decision is not True
    right_child::Union{Node, Nothing} #decision returns True
    prediction::Union{Float64, Nothing} # for leaves
end

# Custom constructor for keyword arguments
function Node(; decision_is_right=nothing, left_child=nothing, right_child=nothing, prediction=nothing)
    Node(decision_is_right, left_child, right_child, prediction)
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
    if tree.decision_is_right(x)
        return tree_prediction(tree.right_child, x)
    else
        return tree_prediction(tree.left_child, x)
    end
end

"""
    lessThan

a basic decision function for testing and playing around
"""
function lessThan(x, threshold::Float64, featureindex::Int =1)::Bool
    return x[featureindex] < threshold
end




export lessThan
export tree_prediction
export Node





end # end the module
