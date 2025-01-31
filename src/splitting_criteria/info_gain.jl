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
    information_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels) -> Float64

This function calculates the information gain for a split in a decision tree. The split is characterized by the partition of the parent_labels into true_child_labels and false_child_labels according to some discriminant function.

# Arguments:
- `parent_labels`: A vector of data labels (e.g., classes or numerical values in the case of regression).
- `true_child_labels`: A vector of a subset of data labels contained in parent_labels.
- `false_child_labels`: A vector of a subset of data labels contained in parent_labels.

# Returns:
- The information gain of the split.
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