# Spliting criterion Information Gain using entropy
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
    probabilities = [occurence / length(labels) for occurence in values(num_occurences)]
    return -sum(p * log2(p) for p in probabilities if p > 0)
end

"""
    information_gain(features::AbstractVector) -> Float64

This function calculates the information gain splitting criterion

# Arguments:
- `parent_labels`: A vector of all considered labels to be split
- `true_child_labels`: A vector of labels of the first splitting part
- `false_child_labels`: A vector of labels of the other splitting part

# Returns:
- The information gain for given split as Float64.
- calculated as follows:
    Information gain = entropy(parent) - [weights] * entropy(children)
"""
function information_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels::AbstractVector) :: Float64
    total = length(parent_labels)

    true_child_weight = length(true_child_labels) / total
    false_child_weight = length(false_child_labels) / total

    true_weighted_entropy = true_child_weight * entropy(true_child_labels)
    false_weighted_entropy = false_child_weight * entropy(false_child_labels)
    weighted_entropy = true_weighted_entropy + false_weighted_entropy

    return entropy(parent_labels) - weighted_entropy
end