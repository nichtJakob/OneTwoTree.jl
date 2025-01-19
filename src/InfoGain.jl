# Spliting criterion Information Gain using entropy
using StatsBase
"""
    entropy(features::AbstractVector) -> Float64

This function calculates the entropy of set of lables

# Arguments:
- `labels`: A vector of labels

# Returns:
- The entropy H(X) = -sum^n_{i=1} P(x_i) log_2(P(x_i))
    with X beeing the labels
    and n the number of elements in X
    and P() beeing the Probability 
"""
function entropy(labels)
    num_occurences = countmap(labels)
    wkeiten = [occurence / length(labels) for occurence in values(num_occurences)]
    return -sum(p * log2(p) for p in wkeiten if p > 0)
end


"""
    information_gain(features::AbstractVector) -> Float64

This function calculates the information gain spliting criterion

# Arguments:
- `parent_labels`: A vector of all considerd labels to be split
- `child_1_labels`: A vector of labels of the first spliting part
- `child_2_labels`: A vector of labels of the other spliting part

# Returns:
- The information gain for given split as Float64. 
- calculated as follows:
    Information gain = entropy(parent) - [weightes] * entropy(children) 
"""
function information_gain(parent_labels::AbstractVector, child_1_labels::AbstractVector, child_2_labels::AbstractVector) :: Float64
    total = length(parent_labels) 

    child_1_weight = length(child_1_labels) / total
    child_2_weight = length(child_2_labels) / total

    weighted_entropy_1 = child_1_weight * entropy(child_1_labels) 
    weighted_entropy_2 = child_2_weight * entropy(child_2_labels) 
    weighted_entropy = weighted_entropy_1 + weighted_entropy_2

    return entropy(parent_labels) - weighted_entropy
end
