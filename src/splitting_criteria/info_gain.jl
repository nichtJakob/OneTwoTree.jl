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
function information_gain(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64}, decision_fn::Function, decision_param::Union{Real, String}, decision_feature::Int64)::Float64

    # Split data in true and false
    split_true, split_false = split_indices(features, node_data, decision_fn, decision_param, decision_feature)

    # Labeling data
    true_labels = labels[split_true]
    false_labels = labels[split_false]
    total_labels = length(labels[node_data])

    if isempty(true_labels) || isempty(false_labels)
        return 0.0
    end

    true_weight = length(true_labels) / total_labels
    false_weight = length(false_labels) / total_labels

    weighted_true_entropy = true_weight * entropy(true_labels)
    weighted_false_entropy = false_weight * entropy(false_labels)
    weighted_entropy = weighted_true_entropy + weighted_false_entropy

    return entropy(labels[node_data]) - weighted_entropy
end

"""
    information_gain(features::AbstractVector) -> Float64

This function calculates the information gain spliting criterion

# Arguments:
- `parent_labels`: A vector of all considerd labels to be split
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